# Ultralytics YOLO, AGPL-3.0 license
"""YOLOE training with an auxiliary RGB-to-depth completion head."""

from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset, YOLOMultiModalDataset
from ultralytics.nn.tasks import YOLOEDepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

from .train import YOLOETrainer
from .val import YOLOEDetectValidator


class DepthSampleMixin:
    """Loads precomputed dense/sparse depth tensors for an existing YOLO sample."""

    def _depth_root(self, key):
        spec = self.data.get(key)
        if spec is None:
            raise ValueError(f"Depth training requires '{key}' in the data yaml.")
        if isinstance(spec, dict):
            spec = spec.get(self.depth_split) or spec.get("train")
        root = Path(spec)
        if not root.is_absolute():
            root = Path(self.data.get("path", ".")) / root
        return root

    def _load_depth(self, key, im_file):
        path = self._depth_root(key) / f"{Path(im_file).stem}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed {key} file: {path}")
        depth = np.load(path).astype(np.float32)
        return np.squeeze(depth)

    @staticmethod
    def _letterbox_depth(depth, out_hw, nearest=False):
        h, w = depth.shape[:2]
        out_h, out_w = out_hw
        r = min(out_h / h, out_w / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if (w, h) != (new_w, new_h):
            interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
            depth = cv2.resize(depth, (new_w, new_h), interpolation=interp)
        dw, dh = out_w - new_w, out_h - new_h
        top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
        left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
        return cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    def add_precomputed_depth(self, label):
        out_hw = label["img"].shape[-2:]
        dense_depth = self._load_depth("dense_depth", label["im_file"])
        sparse_depth = self._load_depth("sparse_depth", label["im_file"])
        sparse_mask = self._load_depth("sparse_mask", label["im_file"])
        valid_mask = self._load_depth("valid_mask", label["im_file"])

        dense_depth = self._letterbox_depth(dense_depth, out_hw, nearest=False)
        sparse_depth = self._letterbox_depth(sparse_depth, out_hw, nearest=True)
        sparse_mask = self._letterbox_depth(sparse_mask, out_hw, nearest=True)
        valid_mask = self._letterbox_depth(valid_mask, out_hw, nearest=True)

        label["dense_depth"] = torch.from_numpy(dense_depth).unsqueeze(0).float()
        label["sparse_depth"] = torch.from_numpy(sparse_depth).unsqueeze(0).float()
        label["sparse_mask"] = torch.from_numpy((sparse_mask > 0).astype(np.float32)).unsqueeze(0)
        label["valid_mask"] = torch.from_numpy((valid_mask > 0).astype(np.float32)).unsqueeze(0)
        return label

    def __getitem__(self, index):
        return self.add_precomputed_depth(super().__getitem__(index))

    @staticmethod
    def collate_fn(batch):
        """Collates YOLO samples and stacks dense/sparse depth tensors."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "texts", "dense_depth", "sparse_depth", "sparse_mask", "valid_mask"}:
                value = torch.stack(value, 0)
            if k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOEDepthDataset(DepthSampleMixin, YOLODataset):
    """YOLO dataset with precomputed depth tensors."""

    def __init__(self, *args, depth_split="train", **kwargs):
        self.depth_split = depth_split
        super().__init__(*args, **kwargs)


class YOLOEDepthMultiModalDataset(DepthSampleMixin, YOLOMultiModalDataset):
    """YOLOE multimodal dataset with precomputed depth tensors."""

    def __init__(self, *args, depth_split="train", **kwargs):
        self.depth_split = depth_split
        super().__init__(*args, **kwargs)


class YOLOEDepthTrainer(YOLOETrainer):
    """YOLOE trainer that keeps detection unchanged and adds an auxiliary depth loss."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEDepthModel initialized with specified config and weights."""
        model = YOLOEDepthModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a YOLOE detection validator while preserving the extra depth loss column."""
        self.loss_names = "box", "cls", "dfl", "depth_abs", "depth_rel", "depth_grad", "depth"
        return YOLOEDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build a YOLOE dataset that returns image, YOLO labels, sparse_depth, valid_mask, and dense_depth.
        """
        if mode == "train" and RANK in {-1, 0}:
            LOGGER.info("Depth training: loading precomputed dense/sparse depth files.")
            unsafe = {
                k: getattr(self.args, k, 0)
                for k in ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr")
            }
            enabled = {k: v for k, v in unsafe.items() if float(v or 0) != 0.0}
            if enabled:
                raise ValueError(
                    "YOLOE depth training requires depth-safe augmentation settings. "
                    f"Disable geometric/mix augmentations; enabled={enabled}"
                )
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = YOLOEDepthMultiModalDataset if mode == "train" else YOLOEDepthDataset
        return dataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            load_vp=self.args.load_vp,
            depth_split=mode,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        target_hw = batch["img"].shape[-2:]
        for key in ("dense_depth", "sparse_depth", "sparse_mask", "valid_mask"):
            value = batch[key].to(self.device, non_blocking=True).float()
            if value.shape[-2:] != target_hw:
                mode = "nearest" if key in {"sparse_depth", "sparse_mask", "valid_mask"} else "bilinear"
                if mode == "bilinear":
                    value = F.interpolate(value, size=target_hw, mode=mode, align_corners=False)
                else:
                    value = F.interpolate(value, size=target_hw, mode=mode)
            batch[key] = value
        return batch
