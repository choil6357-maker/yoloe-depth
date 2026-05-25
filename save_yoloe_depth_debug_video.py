"""Save an RGB/sparse/dense/prediction debug video for the YOLOE depth sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ultralytics.nn.tasks import YOLOEDepthModel  # noqa: F401 - required for torch checkpoint loading
from ultralytics.utils.ops import non_max_suppression

IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/yoloe_convnextv2_spnet_sample/weights/best.pt")
    parser.add_argument("--data", default="tiny_video100/data_depth.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--output", default="runs/yoloe_convnextv2_spnet_sample/debug_depth_prediction.mp4")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--device", default="0")
    parser.add_argument("--text-embeddings", default="runs/yoloe_mobileclip_blt_text/train_label_embeddings.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    return parser.parse_args()


def resolve_device(device):
    if torch.cuda.is_available() and device not in {"", "cpu"}:
        return torch.device(f"cuda:{device}" if device.isdigit() else device)
    return torch.device("cpu")


def split_value(value, split):
    if isinstance(value, dict):
        return value.get(split) or value.get("train")
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def letterbox(array, size, interpolation=cv2.INTER_LINEAR):
    h, w = array.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation) if (w, h) != (new_w, new_h) else array
    dw, dh = size - new_w, size - new_h
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    value = 114 if array.ndim == 3 else 0
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)


def colorize_depth(depth, invalid=None):
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    if invalid is not None:
        colored[invalid] = 0
    return colored


def add_title(image, title):
    cv2.putText(image, title, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, title, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def load_model(weights, device):
    ckpt = torch.load(weights, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        raise ValueError(f"No model found in checkpoint: {weights}")
    model = model.float().to(device).eval()
    return model


def load_text_features(path, names, device):
    path = Path(path)
    if not path.exists():
        print(f"WARNING: text embeddings not found, bbox logits will use zeros: {path}")
        return torch.zeros(1, len(names), 512, device=device)

    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        features = []
        for _, name in names.items():
            if name in data:
                features.append(data[name])
            else:
                features.append(next(iter(data.values())))
        features = torch.stack(features, 0)
    else:
        features = data
        if features.ndim == 1:
            features = features.unsqueeze(0)
    return features.unsqueeze(0).to(device=device, dtype=torch.float32)


def label_file_for_image(root, image_file):
    rel = image_file.relative_to(root)
    parts = list(rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


def load_gt_boxes(root, image_file, image_shape, size):
    label_file = label_file_for_image(root, image_file)
    if not label_file.exists():
        return []

    h, w = image_shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    left = int(round((size - new_w) / 2 - 0.1))
    top = int(round((size - new_h) / 2 - 0.1))
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = map(float, line.split()[:5])
        x1 = (xc - bw / 2) * w * r + left
        y1 = (yc - bh / 2) * h * r + top
        x2 = (xc + bw / 2) * w * r + left
        y2 = (yc + bh / 2) * h * r + top
        boxes.append((x1, y1, x2, y2, int(cls)))
    return boxes


def draw_boxes(image, detections, gt_boxes, names):
    out = image.copy()
    for x1, y1, x2, y2, cls in gt_boxes:
        label = f"GT {names.get(cls, cls)}"
        p1 = int(round(x1)), int(round(y1))
        p2 = int(round(x2)), int(round(y2))
        cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
        text_y = min(max(p2[1] + 14, 14), out.shape[0] - 4)
        cv2.putText(out, label, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)
        label = f"P {names.get(cls, cls)} {conf:.2f}"
        p1 = int(round(x1)), int(round(y1))
        p2 = int(round(x2)), int(round(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        text_y = max(p1[1] - 5, 12)
        cv2.putText(out, label, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    data_path = Path(args.data)
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_path.parent))
    image_root = resolve_path(root, split_value(data[args.split], args.split))
    dense_root = resolve_path(root, split_value(data["dense_depth"], args.split))
    sparse_root = resolve_path(root, split_value(data["sparse_depth"], args.split))
    valid_root = resolve_path(root, split_value(data["valid_mask"], args.split))
    names = {int(k): v for k, v in data.get("names", {0: "object"}).items()}
    images = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.max_frames > 0:
        images = images[: args.max_frames]
    if not images:
        raise FileNotFoundError(f"No images found under {image_root}")

    device = resolve_device(args.device)
    model = load_model(args.weights, device)
    tpe = load_text_features(args.text_embeddings, names, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (args.panel_size * 5, args.panel_size)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    try:
        for image_file in images:
            image = cv2.imread(str(image_file))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_file}")
            dense = np.load(dense_root / f"{image_file.stem}.npy").astype(np.float32)
            sparse = np.load(sparse_root / f"{image_file.stem}.npy").astype(np.float32)
            valid = np.load(valid_root / f"{image_file.stem}.npy").astype(np.float32)

            gt_boxes = load_gt_boxes(root, image_file, image.shape, args.imgsz)
            image_in = letterbox(image, args.imgsz, cv2.INTER_LINEAR)
            dense_in = letterbox(np.squeeze(dense), args.imgsz, cv2.INTER_LINEAR)
            sparse_in = letterbox(np.squeeze(sparse), args.imgsz, cv2.INTER_NEAREST)
            valid_in = letterbox(np.squeeze(valid), args.imgsz, cv2.INTER_NEAREST) > 0

            rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255.0
            sparse_tensor = torch.from_numpy(sparse_in).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad():
                det, pred = model(tensor, tpe=tpe, return_depth=True, sparse_depth=sparse_tensor)
            pred = pred[0, 0].detach().cpu().numpy()
            depth_mae = float(np.abs(pred - dense_in)[valid_in].mean())
            det = det[0] if isinstance(det, tuple) else det
            detections = non_max_suppression(
                det,
                conf_thres=args.conf,
                iou_thres=args.iou,
                max_det=args.max_det,
                nc=len(names),
            )[0].detach().cpu()

            panels = [
                (image_in, "RGB"),
                (draw_boxes(image_in, detections, gt_boxes, names), f"BBox P{len(detections)}/G{len(gt_boxes)}"),
                (colorize_depth(sparse_in, invalid=sparse_in <= 0), "Sparse"),
                (colorize_depth(dense_in, invalid=~valid_in), "Dense"),
                (colorize_depth(pred, invalid=~valid_in), f"Pred MAE {depth_mae:.3f}"),
            ]
            rendered = []
            for panel, title in panels:
                panel = cv2.resize(panel, (args.panel_size, args.panel_size), interpolation=cv2.INTER_NEAREST)
                rendered.append(add_title(panel, title))
            writer.write(np.concatenate(rendered, axis=1))
    finally:
        writer.release()

    print(f"Saved depth debug video: {output}")


if __name__ == "__main__":
    main()
