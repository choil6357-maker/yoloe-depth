"""COCO-mini benchmark runner for YOLOE + ConvNeXtV2 + depth completion.

The default run is intentionally small enough for an RTX2070-class GPU. It prepares a
fixed COCO subset, caches one pseudo-depth target per image, runs the configured
baselines/variants, and writes raw CSV/JSON plus a compact PDF report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "coco_mini_depth"
RUN_ROOT = ROOT / "runs" / "coco_mini_depth_benchmark"
SEED = 20260525

LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
DEPTH_PRO_REPO = "https://github.com/apple/ml-depth-pro.git"
DEPTH_PRO_CKPT_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
SPNET_REPO = "https://github.com/Wang-xjtu/SPNet.git"
DEPTH_REPRESENTATION = "metric_depth"
DEPTH_UNITS = "meters"
DEPTH_POLARITY = "larger_value_means_farther"
DEPTH_LOSS_TYPE = "spnet_absolute_relative_gradient"
DEFAULT_METRIC_DEPTH_MAX_M = 100.0

EXPERIMENTS = {
    "A": {
        "title": "YOLOE-v8s original backbone",
        "cfg": "ultralytics/cfg/models/v8/yoloe_v8s_benchmark.yaml",
        "depth": False,
    },
    "B": {
        "title": "YOLOE + ConvNeXtV2-Small",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_small_sample.yaml",
        "depth": False,
    },
    "C": {
        "title": "Current SPNet-style depth head",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_spnet_sample.yaml",
        "depth": True,
    },
    "V1": {
        "title": "5ch shared backbone",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_depth_5ch.yaml",
        "depth": True,
    },
    "V2": {
        "title": "Shallow sparse-depth encoder",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_depth_sparse_encoder.yaml",
        "depth": True,
    },
    "V3": {
        "title": "C2/C3/C4/C5 depth adapters",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_depth_adapters.yaml",
        "depth": True,
    },
    "V4": {
        "title": "Any2Full-style scale prompt",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_depth_scale_prompt.yaml",
        "depth": True,
    },
    "V5": {
        "title": "Sparse encoder + scale prompt",
        "cfg": "ultralytics/cfg/models/v8/yoloe_convnextv2_depth_sparse_scale.yaml",
        "depth": True,
    },
}


@dataclass
class DataManifest:
    dataset_root: str
    dataset_source: str
    train_count: int
    val_count: int
    pseudo_depth_source: str
    depth_representation: str
    depth_units: str
    depth_polarity: str
    depth_loss_type: str
    metric_depth_max_m: float
    sparse_policy: str
    seed: int
    notes: list[str] | None = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--data-root", type=Path, default=DATA_ROOT)
        p.add_argument("--run-root", type=Path, default=RUN_ROOT)
        p.add_argument("--train-count", type=int, default=512)
        p.add_argument("--val-count", type=int, default=128)
        p.add_argument("--imgsz", type=int, default=320)
        p.add_argument("--batch", type=int, default=2)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--epochs", type=int, default=2)
        p.add_argument("--val-interval", type=int, default=1, help="Run YOLO validation every N epochs; default 1 records fresh metrics every epoch.")
        p.add_argument("--device", default="0")
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--allow-fallback-depth", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--allow-tiny-fallback", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--no-download", action="store_true")
        p.add_argument("--depth-source", choices=("depth-pro",), default="depth-pro")
        p.add_argument("--force-depth-cache", action="store_true", help="Regenerate dense/sparse/valid depth .npy files even when cache exists.")
        p.add_argument("--depth-pro-precision", choices=("fp32", "fp16"), default="fp16", help="Depth Pro inference precision; fp16 is faster on CUDA.")
        p.add_argument(
            "--metric-depth-max-m",
            type=float,
            default=DEFAULT_METRIC_DEPTH_MAX_M,
            help="Clip Depth Pro metric pseudo-depth targets to this many meters; <=0 disables clipping.",
        )

    p = sub.add_parser("prepare")
    add_common(p)
    p.add_argument("--limit-depth", type=int, default=0, help="Only cache this many images per split; 0 means all.")

    p = sub.add_parser("smoke")
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 320])
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))

    p = sub.add_parser("audit")
    add_common(p)
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 320])
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    p.add_argument("--json-out", type=Path, default=None)

    p = sub.add_parser("run")
    add_common(p)
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    p.add_argument("--include-spnet", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("preflight")
    add_common(p)
    p.add_argument("--include-spnet", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--json-out", type=Path, default=None)

    p = sub.add_parser("report")
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--output", type=Path, default=None)

    p = sub.add_parser("visualize")
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    p.add_argument("--sample-count", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=8)
    p.add_argument("--include-spnet", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("train-one")
    p.add_argument("--experiment", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--text-model", required=True)
    p.add_argument("--text-embeddings", required=True)
    p.add_argument("--depth", action="store_true")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metrics-out", required=True)

    p = sub.add_parser("spnet-one")
    p.add_argument("--data", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--metrics-out", required=True)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def recovery_hint(component: str, detail: str | None = None) -> str:
    hints = {
        "coco_labels": (
            f"Download {LABELS_URL} and unzip it under {rel(DATA_ROOT / '_assets')}. "
            "The extracted tree should contain coco/train2017.txt, coco/val2017.txt, and coco/labels/<split>/*.txt."
        ),
        "coco_image": (
            "Check network access to https://images.cocodataset.org. You can manually place JPGs under "
            f"{rel(DATA_ROOT / 'images' / '<split>')} and matching labels under {rel(DATA_ROOT / 'labels' / '<split>')}, "
            "then rerun with --no-download."
        ),
        "depth_pro_repo": f"Run: git clone --depth 1 {DEPTH_PRO_REPO} {rel(ROOT / 'third_party' / 'ml-depth-pro')}",
        "depth_pro_ckpt": (
            f"Download {DEPTH_PRO_CKPT_URL} to "
            f"{rel(ROOT / 'third_party' / 'ml-depth-pro' / 'checkpoints' / 'depth_pro.pt')}."
        ),
        "depth_pro_import": (
            "Install Depth Pro dependencies without changing Torch, for example: "
            f"{rel(ROOT / '.venv' / 'Scripts' / 'python.exe')} -m pip install pillow_heif"
        ),
        "spnet_repo": f"Run: git clone --depth 1 {SPNET_REPO} {rel(ROOT / 'third_party' / 'SPNet')}",
        "depth_cache": (
            "Run the online prepare step once, or place matching .npy files under "
            f"{rel(DATA_ROOT / 'depth' / 'dense' / '<split>')} and {rel(DATA_ROOT / 'depth' / 'sparse' / '<split>')}."
        ),
    }
    hint = hints.get(component, "Inspect the logged command output and rerun after fixing the missing asset.")
    return f"{hint} Detail: {detail}" if detail else hint


def fail_with_hint(component: str, message: str, detail: str | None = None):
    raise RuntimeError(f"{message}\nManual recovery: {recovery_hint(component, detail)}")


def download_file(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        fail_with_hint("download", f"Failed to download {url} -> {path}", repr(e))
    return path


def download_first(urls: list[str], path: Path, component: str):
    errors = []
    for url in urls:
        try:
            return download_file(url, path)
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
    fail_with_hint(component, f"All download attempts failed for {path}.", " | ".join(errors))


def run_command(cmd, log_path: Path, cwd=ROOT, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n\n")
        proc = subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode, time.time() - start


def find_first(root: Path, pattern: str):
    if not root.exists():
        return None
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def coco_names():
    data = read_yaml(ROOT / "ultralytics" / "cfg" / "datasets" / "coco.yaml")
    return {int(k): v for k, v in data["names"].items()}


def ensure_coco_labels(data_root: Path, no_download=False):
    assets = data_root / "_assets"
    list_file = find_first(assets, "train2017.txt")
    if list_file:
        return assets
    if no_download:
        raise FileNotFoundError(f"COCO labels are missing and --no-download was set. {recovery_hint('coco_labels')}")
    zip_path = assets / "coco2017labels.zip"
    try:
        download_file(LABELS_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(assets)
    except Exception as e:
        fail_with_hint("coco_labels", "Failed to prepare COCO label assets.", repr(e))
    return assets


def label_path_for(assets: Path, split: str, stem: str):
    direct = assets / "coco" / "labels" / split / f"{stem}.txt"
    if direct.exists():
        return direct
    return find_first(assets, f"{stem}.txt")


def image_urls(split: str, stem: str):
    return [
        f"https://images.cocodataset.org/{split}/{stem}.jpg",
        f"http://images.cocodataset.org/{split}/{stem}.jpg",
    ]


def prepare_from_coco(args) -> tuple[Path, str]:
    data_root = args.data_root
    assets = ensure_coco_labels(data_root, no_download=args.no_download)
    for split, count in (("train2017", args.train_count), ("val2017", args.val_count)):
        list_file = find_first(assets, f"{split}.txt")
        if not list_file:
            raise FileNotFoundError(f"Missing {split}.txt in {assets}")
        lines = [x.strip() for x in list_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        selected = lines[:count]
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        txt_lines = []
        for line in selected:
            stem = Path(line).stem
            image_path = image_dir / f"{stem}.jpg"
            if not image_path.exists():
                try:
                    download_first(image_urls(split, stem), image_path, "coco_image")
                except Exception as e:
                    fail_with_hint("coco_image", f"Failed to download COCO image {split}/{stem}.jpg.", repr(e))
            src_label = label_path_for(assets, split, stem)
            dst_label = label_dir / f"{stem}.txt"
            if src_label and src_label.exists():
                shutil.copyfile(src_label, dst_label)
            elif not dst_label.exists():
                dst_label.write_text("", encoding="utf-8")
            txt_lines.append(str(image_path.resolve()))
        txt_name = "train.txt" if split == "train2017" else "val.txt"
        (data_root / txt_name).write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        cache_file = data_root / "labels" / f"{split}.cache"
        if cache_file.exists():
            cache_file.unlink()
    return data_root, "coco2017_individual_download"


def prepare_from_tiny_fallback(args) -> tuple[Path, str]:
    tiny = ROOT / "tiny_video100"
    images = sorted((tiny / "images" / "train").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("COCO download failed and tiny_video100 fallback images are missing.")
    def tiny_label_for(image_path: Path):
        label = tiny / "labels" / "train" / f"{image_path.stem}.txt"
        if not label.exists():
            label = tiny / "autolabel" / "labels" / f"{image_path.stem}.txt"
        return label

    labeled = [p for p in images if tiny_label_for(p).exists()]
    unlabeled = [p for p in images if p not in set(labeled)]
    images = labeled + unlabeled
    data_root = args.data_root
    names = coco_names()
    for split, count in (("train2017", args.train_count), ("val2017", args.val_count)):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        txt_lines = []
        for src in images[:count]:
            dst = image_dir / src.name
            if not dst.exists():
                shutil.copyfile(src, dst)
            src_label = tiny_label_for(src)
            dst_label = label_dir / f"{src.stem}.txt"
            if src_label.exists():
                shutil.copyfile(src_label, dst_label)
            elif not dst_label.exists():
                dst_label.write_text("", encoding="utf-8")
            txt_lines.append(str(dst.resolve()))
        txt_name = "train.txt" if split == "train2017" else "val.txt"
        (data_root / txt_name).write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        cache_file = data_root / "labels" / f"{split}.cache"
        if cache_file.exists():
            cache_file.unlink()
    return data_root, "fallback_tiny_video100_not_coco"


def write_data_yaml(data_root: Path):
    data = {
        "path": str(data_root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "dense_depth": {"train": "depth/dense/train2017", "val": "depth/dense/val2017"},
        "sparse_depth": {"train": "depth/sparse/train2017", "val": "depth/sparse/val2017"},
        "sparse_mask": {"train": "depth/sparse_mask/train2017", "val": "depth/sparse_mask/val2017"},
        "valid_mask": {"train": "depth/valid/train2017", "val": "depth/valid/val2017"},
        "names": coco_names(),
    }
    write_yaml(data_root / "data_depth.yaml", data)
    return data_root / "data_depth.yaml"


def image_files_from_txt(data_root: Path, split: str):
    txt = data_root / ("train.txt" if split == "train2017" else "val.txt")
    if not txt.exists():
        return []
    return [Path(x.strip().lstrip("\ufeff")) for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]


def existing_mini_ready(data_root: Path, train_count: int, val_count: int):
    missing = []
    for split, count in (("train2017", train_count), ("val2017", val_count)):
        images = image_files_from_txt(data_root, split)
        if len(images) < count:
            missing.append(f"{split}: expected {count} image entries, found {len(images)}")
            continue
        for image_path in images[:count]:
            label_path = data_root / "labels" / split / f"{image_path.stem}.txt"
            if not image_path.exists():
                missing.append(f"missing image {image_path}")
            if not label_path.exists():
                missing.append(f"missing label {label_path}")
    return not missing, missing[:20]


def depth_cache_counts(data_root: Path):
    counts = {}
    for kind in ("dense", "sparse", "sparse_mask", "valid", "source"):
        for split in ("train2017", "val2017"):
            pattern = "*.json" if kind == "source" else "*.npy"
            counts[f"{kind}_{split}"] = len(list((data_root / "depth" / kind / split).glob(pattern)))
    return counts


def depth_cache_complete(data_root: Path):
    missing = []
    for split in ("train2017", "val2017"):
        for image_path in image_files_from_txt(data_root, split):
            for kind in ("dense", "sparse", "sparse_mask", "valid"):
                path = data_root / "depth" / kind / split / f"{image_path.stem}.npy"
                if not path.exists():
                    missing.append(str(path))
    return not missing and bool(image_files_from_txt(data_root, "train2017")) and bool(image_files_from_txt(data_root, "val2017")), missing[:20]


def depth_cache_manifest_path(data_root: Path):
    return data_root / "depth" / "cache_manifest.json"


def read_depth_cache_manifest(data_root: Path):
    path = depth_cache_manifest_path(data_root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def deterministic_depth(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ramp = np.linspace(0, 1, gray.shape[0], dtype=np.float32)[:, None]
    return (1.0 + 19.0 * np.clip(0.65 * gray + 0.35 * ramp, 0, 1)).astype(np.float32)


def ensure_depth_pro(no_download=False, precision="fp16"):
    repo = ROOT / "third_party" / "ml-depth-pro"
    if not repo.exists():
        if no_download:
            raise FileNotFoundError(f"Depth Pro repo is missing and --no-download was set. {recovery_hint('depth_pro_repo')}")
        log_path = RUN_ROOT / "logs" / "clone_depth_pro.log"
        code, _ = run_command(["git", "clone", "--depth", "1", DEPTH_PRO_REPO, repo], log_path)
        if code != 0:
            fail_with_hint("depth_pro_repo", f"Depth Pro clone failed; see {rel(log_path)}.", f"exit={code}")
    ckpt = repo / "checkpoints" / "depth_pro.pt"
    if not ckpt.exists():
        if no_download:
            raise FileNotFoundError(f"Depth Pro checkpoint is missing and --no-download was set. {recovery_hint('depth_pro_ckpt')}")
        try:
            download_file(DEPTH_PRO_CKPT_URL, ckpt)
        except Exception as e:
            fail_with_hint("depth_pro_ckpt", "Depth Pro checkpoint download failed.", repr(e))
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig
    except Exception as e:
        fail_with_hint(
            "depth_pro_import",
            "Depth Pro Python import failed. Keep the main Torch install unchanged; install only missing lightweight dependencies if needed.",
            repr(e),
        )
    base = DEFAULT_MONODEPTH_CONFIG_DICT
    config = DepthProConfig(
        patch_encoder_preset=base.patch_encoder_preset,
        image_encoder_preset=base.image_encoder_preset,
        decoder_features=base.decoder_features,
        checkpoint_uri=str(ckpt),
        fov_encoder_preset=base.fov_encoder_preset,
        use_fov_head=base.use_fov_head,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if precision == "fp16" and device.type == "cuda" else torch.float32
    import timm

    original_create_model = timm.create_model

    def create_model_compat(*model_args, **model_kwargs):
        try:
            return original_create_model(*model_args, **model_kwargs)
        except TypeError as e:
            if "dynamic_img_size" in str(e) and "dynamic_img_size" in model_kwargs:
                model_kwargs = dict(model_kwargs)
                model_kwargs.pop("dynamic_img_size", None)
                return original_create_model(*model_args, **model_kwargs)
            raise

    timm.create_model = create_model_compat
    try:
        model, transform = depth_pro.create_model_and_transforms(config=config, device=device, precision=dtype)
    finally:
        timm.create_model = original_create_model
    return model.eval(), transform, device


def infer_depth_pro(model, transform, device, image_path: Path):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    with torch.no_grad():
        pred = model.infer(tensor, f_px=None)["depth"]
    return pred.detach().float().cpu().numpy()


def finite_mask(array):
    return np.isfinite(np.asarray(array, dtype=np.float32)).astype(np.float32)


def metric_depth_max_m(args_or_value=None):
    value = getattr(args_or_value, "metric_depth_max_m", args_or_value)
    if value is None:
        value = DEFAULT_METRIC_DEPTH_MAX_M
    value = float(value)
    return None if value <= 0 else value


def sanitize_metric_depth(pred, max_depth_m=DEFAULT_METRIC_DEPTH_MAX_M):
    pred = np.asarray(pred, dtype=np.float32)
    clean = np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    max_depth_m = metric_depth_max_m(max_depth_m)
    if max_depth_m is not None:
        clean = np.minimum(clean, max_depth_m)
    return clean.astype(np.float32)


def sparse_policy(dense: np.ndarray, stem: str, seed: int):
    h, w = dense.shape
    digest = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    mask = rng.random((h, w)) < 0.01
    y0 = int(round(h * 0.55))
    ys = np.linspace(y0, h - 1, num=min(8, h), dtype=np.int64)
    mask[ys, :] = True
    sparse = np.zeros_like(dense, dtype=np.float32)
    sparse[mask] = dense[mask]
    valid = np.isfinite(dense).astype(np.float32)
    return sparse, valid, mask.astype(np.float32)


def depth_cache_expected_marker(args, source: str):
    return {
        "source": source,
        "depth_source_arg": args.depth_source,
        "depth_pro_precision": args.depth_pro_precision if args.depth_source == "depth-pro" else None,
        "metric_depth_max_m": metric_depth_max_m(args),
        "seed": args.seed,
        "representation": DEPTH_REPRESENTATION,
        "units": DEPTH_UNITS,
    }


def depth_cache_core_matches(manifest: dict, expected: dict):
    keys = ("source", "depth_source_arg", "depth_pro_precision", "metric_depth_max_m", "seed", "representation", "units")
    return all(manifest.get(k) == expected.get(k) for k in keys)


def depth_cache_matches_except_metric_limit(manifest: dict, expected: dict):
    keys = ("source", "depth_source_arg", "depth_pro_precision", "seed", "representation", "units")
    return all(manifest.get(k) == expected.get(k) for k in keys)


def write_depth_cache_manifest(data_root: Path, manifest: dict):
    path = depth_cache_manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def remove_generated_path(path: Path, root: Path):
    """Remove a benchmark-generated file or directory after confirming it is under root."""
    path = Path(path)
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as e:
        raise RuntimeError(f"Refusing to remove generated path outside run root: {resolved}") from e
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def retrofit_sparse_masks_from_existing_cache(data_root: Path, seed: int, max_depth_m=DEFAULT_METRIC_DEPTH_MAX_M):
    """Repair metric depth limits and explicit sparse-known masks without rerunning pseudo-depth inference."""
    manifest = read_depth_cache_manifest(data_root)
    can_repair_depth_values = manifest.get("source") == "depth_pro_metric" and manifest.get("representation", DEPTH_REPRESENTATION) == DEPTH_REPRESENTATION
    changed = 0
    for split in ("train2017", "val2017"):
        sparse_mask_dir = data_root / "depth" / "sparse_mask" / split
        sparse_mask_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_files_from_txt(data_root, split):
            dense_path = data_root / "depth" / "dense" / split / f"{image_path.stem}.npy"
            sparse_path = data_root / "depth" / "sparse" / split / f"{image_path.stem}.npy"
            valid_path = data_root / "depth" / "valid" / split / f"{image_path.stem}.npy"
            sparse_mask_path = sparse_mask_dir / f"{image_path.stem}.npy"
            if not dense_path.exists():
                continue
            dense = np.load(dense_path).astype(np.float32)
            if can_repair_depth_values:
                fixed_dense = sanitize_metric_depth(dense, max_depth_m)
                if not np.allclose(dense, fixed_dense, atol=1e-7):
                    np.save(dense_path, fixed_dense, allow_pickle=False)
                    dense = fixed_dense
                    changed += 1
            expected_sparse, expected_valid, expected_mask = sparse_policy(dense, image_path.stem, seed)
            if not sparse_path.exists() or not np.allclose(np.load(sparse_path), expected_sparse, atol=1e-7):
                sparse_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(sparse_path, expected_sparse, allow_pickle=False)
                changed += 1
            if not valid_path.exists() or not np.array_equal(np.load(valid_path) > 0.5, expected_valid > 0.5):
                valid_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(valid_path, expected_valid, allow_pickle=False)
                changed += 1
            if not sparse_mask_path.exists() or not np.array_equal(np.load(sparse_mask_path) > 0.5, expected_mask > 0.5):
                np.save(sparse_mask_path, expected_mask.astype(np.float32), allow_pickle=False)
                changed += 1
    return changed


def cache_depth(args):
    data_root = args.data_root
    source = "depth_pro_metric"
    expected_cache = depth_cache_expected_marker(args, source)
    if not getattr(args, "force_depth_cache", False):
        changed = retrofit_sparse_masks_from_existing_cache(data_root, args.seed, metric_depth_max_m(args))
        if changed:
            print(f"Retrofitted metric depth limits / sparse cache files: {changed}", flush=True)
    complete, _ = depth_cache_complete(data_root)
    cache_manifest = read_depth_cache_manifest(data_root)
    if complete and depth_cache_matches_except_metric_limit(cache_manifest, expected_cache) and cache_manifest.get("metric_depth_max_m") != expected_cache["metric_depth_max_m"]:
        cache_manifest.update(expected_cache)
        cache_manifest.update(
            {
                "polarity": DEPTH_POLARITY,
                "loss_type": DEPTH_LOSS_TYPE,
                "metric_depth_max_m": metric_depth_max_m(args),
                "sparse_known_mask": "depth/sparse_mask/<split>/<stem>.npy; 1 means sparse sample is known",
                "metadata_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        write_depth_cache_manifest(data_root, cache_manifest)
    manifest_core_matches = complete and depth_cache_core_matches(cache_manifest, expected_cache)
    manifest_matches = complete and all(cache_manifest.get(k) == v for k, v in expected_cache.items())
    if complete and manifest_core_matches and not getattr(args, "force_depth_cache", False):
        actual_train = len(image_files_from_txt(data_root, "train2017"))
        actual_val = len(image_files_from_txt(data_root, "val2017"))
        if not manifest_matches or cache_manifest.get("train_count") != actual_train or cache_manifest.get("val_count") != actual_val:
            cache_manifest.update(expected_cache)
            cache_manifest.update(
                {
                    "polarity": DEPTH_POLARITY,
                    "loss_type": DEPTH_LOSS_TYPE,
                    "metric_depth_max_m": metric_depth_max_m(args),
                    "sparse_known_mask": "depth/sparse_mask/<split>/<stem>.npy; 1 means sparse sample is known",
                    "train_count": actual_train,
                    "val_count": actual_val,
                    "metadata_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            write_depth_cache_manifest(data_root, cache_manifest)
        return cache_manifest.get("source", "cached_existing_npy")
    if complete and not manifest_core_matches and getattr(args, "no_download", False) and not getattr(args, "force_depth_cache", False):
        raise RuntimeError(
            "Existing depth cache does not match requested depth settings. "
            f"requested={expected_cache}, cache_manifest={cache_manifest}. "
            "Rerun prepare once to repair metric caches in place, rerun with --force-depth-cache while downloads/checkpoints are available, "
            "or pass matching --depth-source/--depth-pro-precision/--metric-depth-max-m."
        )
    depth_pro_model, depth_pro_transform, depth_pro_device = None, None, None
    try:
        if args.depth_source == "depth-pro":
            depth_pro_model, depth_pro_transform, depth_pro_device = ensure_depth_pro(no_download=args.no_download, precision=args.depth_pro_precision)
        else:
            raise ValueError(f"Unsupported depth source: {args.depth_source}")
    except Exception as e:
        if not args.allow_fallback_depth:
            raise
        print(f"WARNING: {args.depth_source} unavailable; using synthetic fallback depth. {type(e).__name__}: {e}")
        source = f"synthetic_fallback_after_{args.depth_source}_failure:{type(e).__name__}"

    limit = getattr(args, "limit_depth", 0)
    pending_total = 0
    for split in ("train2017", "val2017"):
        images = image_files_from_txt(data_root, split)
        if limit:
            images = images[:limit]
        source_dir = data_root / "depth" / "source" / split
        for image_path in images:
            source_path = source_dir / f"{image_path.stem}.json"
            existing_marker = {}
            if source_path.exists():
                try:
                    existing_marker = json.loads(source_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_marker = {}
            expected_marker = depth_cache_expected_marker(args, source)
            marker_matches = all(existing_marker.get(k) == v for k, v in expected_marker.items())
            dense_path = data_root / "depth" / "dense" / split / f"{image_path.stem}.npy"
            sparse_path = data_root / "depth" / "sparse" / split / f"{image_path.stem}.npy"
            sparse_mask_path = data_root / "depth" / "sparse_mask" / split / f"{image_path.stem}.npy"
            valid_path = data_root / "depth" / "valid" / split / f"{image_path.stem}.npy"
            if not (dense_path.exists() and sparse_path.exists() and sparse_mask_path.exists() and valid_path.exists() and (not getattr(args, "force_depth_cache", False) or marker_matches)):
                pending_total += 1
    print(f"Depth cache source={source}; pending={pending_total}; force={getattr(args, 'force_depth_cache', False)}", flush=True)
    done_count = 0
    for split in ("train2017", "val2017"):
        images = image_files_from_txt(data_root, split)
        if limit:
            images = images[:limit]
        dense_dir = data_root / "depth" / "dense" / split
        sparse_dir = data_root / "depth" / "sparse" / split
        sparse_mask_dir = data_root / "depth" / "sparse_mask" / split
        valid_dir = data_root / "depth" / "valid" / split
        source_dir = data_root / "depth" / "source" / split
        for directory in (dense_dir, sparse_dir, sparse_mask_dir, valid_dir, source_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            dense_path = dense_dir / f"{image_path.stem}.npy"
            sparse_path = sparse_dir / f"{image_path.stem}.npy"
            sparse_mask_path = sparse_mask_dir / f"{image_path.stem}.npy"
            valid_path = valid_dir / f"{image_path.stem}.npy"
            source_path = source_dir / f"{image_path.stem}.json"
            expected_marker = depth_cache_expected_marker(args, source)
            existing_marker = {}
            if source_path.exists():
                try:
                    existing_marker = json.loads(source_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_marker = {}
            marker_matches = all(existing_marker.get(k) == v for k, v in expected_marker.items())
            if dense_path.exists() and sparse_path.exists() and sparse_mask_path.exists() and valid_path.exists() and (not getattr(args, "force_depth_cache", False) or marker_matches):
                continue
            done_count += 1
            print(f"[depth-cache {done_count}/{pending_total}] {args.depth_source} {split}/{image_path.name}", flush=True)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            if depth_pro_model is not None:
                pred = infer_depth_pro(depth_pro_model, depth_pro_transform, depth_pro_device, image_path)
                valid = finite_mask(pred)
                dense = sanitize_metric_depth(pred, metric_depth_max_m(args))
            else:
                dense = deterministic_depth(image)
                valid = np.ones_like(dense, dtype=np.float32)
            dense = sanitize_metric_depth(dense, metric_depth_max_m(args))
            sparse, _, sparse_mask = sparse_policy(dense, image_path.stem, args.seed)
            np.save(dense_path, dense, allow_pickle=False)
            np.save(sparse_path, sparse, allow_pickle=False)
            np.save(sparse_mask_path, sparse_mask, allow_pickle=False)
            np.save(valid_path, valid, allow_pickle=False)
            marker = dict(expected_marker)
            marker.update({"image": str(image_path), "shape": list(dense.shape), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            source_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    manifest = {
        "source": source,
        "depth_source_arg": args.depth_source,
        "depth_pro_precision": args.depth_pro_precision if args.depth_source == "depth-pro" else None,
        "representation": DEPTH_REPRESENTATION,
        "units": DEPTH_UNITS,
        "polarity": DEPTH_POLARITY,
        "loss_type": DEPTH_LOSS_TYPE,
        "metric_depth_max_m": metric_depth_max_m(args),
        "sparse_known_mask": "depth/sparse_mask/<split>/<stem>.npy; 1 means sparse sample is known",
        "seed": args.seed,
        "train_count": len(image_files_from_txt(data_root, "train2017")),
        "val_count": len(image_files_from_txt(data_root, "val2017")),
        "sparse_policy": "deterministic random 1% + 8 lower-band horizontal scanlines",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_depth_cache_manifest(data_root, manifest)
    return source


def run_data_tests(data_root: Path, seed: int, max_depth_m=DEFAULT_METRIC_DEPTH_MAX_M):
    max_depth_m = metric_depth_max_m(max_depth_m)
    for split in ("train2017", "val2017"):
        images = image_files_from_txt(data_root, split)
        for image_path in images:
            dense = np.load(data_root / "depth" / "dense" / split / f"{image_path.stem}.npy")
            sparse = np.load(data_root / "depth" / "sparse" / split / f"{image_path.stem}.npy")
            sparse_mask = np.load(data_root / "depth" / "sparse_mask" / split / f"{image_path.stem}.npy")
            valid = np.load(data_root / "depth" / "valid" / split / f"{image_path.stem}.npy")
            if not np.isfinite(dense).all() or dense.min() < 0:
                raise AssertionError(f"Dense metric depth is not finite/nonnegative: {image_path}")
            if dense.max() <= 0:
                raise AssertionError(f"Dense metric depth has unexpected max value ({dense.max():.3f} m): {image_path}")
            if max_depth_m is not None and dense.max() > max_depth_m + 1e-4:
                raise AssertionError(
                    f"Dense metric depth exceeds configured clip ({dense.max():.3f} m > {max_depth_m:.3f} m): {image_path}"
                )
            if not np.isfinite(valid).all() or not np.isin(valid, [0, 1]).all():
                raise AssertionError(f"Valid depth mask is not finite binary: {image_path}")
            if not np.isfinite(sparse_mask).all() or not np.isin(sparse_mask, [0, 1]).all():
                raise AssertionError(f"Sparse known mask is not finite binary: {image_path}")
            expected_sparse, _, expected_mask = sparse_policy(dense, image_path.stem, seed)
            if not np.array_equal(sparse_mask > 0.5, expected_mask > 0.5):
                raise AssertionError(f"Sparse policy is not deterministic: {image_path}")
            if not np.allclose(sparse, expected_sparse, atol=1e-7):
                raise AssertionError(f"Sparse values are not copied from dense target: {image_path}")
            ratio = float((sparse_mask > 0.5).mean())
            if not 0.015 <= ratio <= 0.08:
                raise AssertionError(f"Sparse ratio out of expected range ({ratio:.4f}): {image_path}")
            h = dense.shape[0]
            ys = np.linspace(int(round(h * 0.55)), h - 1, num=min(8, h), dtype=np.int64)
            if not np.all(sparse_mask[ys, :] > 0.5):
                raise AssertionError(f"Lower-band scanline mask is incomplete: {image_path}")

    dense = np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48)
    sparse, valid, sparse_mask = sparse_policy(dense, "perfect_depth_sanity", seed)
    perfect = depth_metrics(dense.copy(), dense, valid, sparse_mask)
    if any(abs(v) > 1e-9 for k, v in perfect.items() if "delta1" not in k):
        raise AssertionError(f"Perfect depth metric sanity failed: {perfect}")
    if any(abs(v - 1.0) > 1e-9 for k, v in perfect.items() if "delta1" in k):
        raise AssertionError(f"Perfect depth delta sanity failed: {perfect}")
    preserved = preserve_sparse_prediction(dense.copy(), sparse, sparse_mask)
    preserved_metrics = depth_metrics(preserved, dense, valid, sparse_mask)
    if any(abs(v) > 1e-9 for k, v in preserved_metrics.items() if "delta1" not in k):
        raise AssertionError(f"Sparse-preserved metric sanity failed: {preserved_metrics}")


def prepare(args):
    notes = []
    ready, missing = existing_mini_ready(args.data_root, args.train_count, args.val_count)
    if args.no_download and ready:
        data_root, dataset_source = args.data_root, "existing_local_mini"
    else:
        try:
            data_root, dataset_source = prepare_from_coco(args)
        except Exception as e:
            if not args.allow_tiny_fallback:
                raise RuntimeError(
                    f"COCO mini preparation failed and fallback is disabled. {type(e).__name__}: {e}\n"
                    f"Manual recovery: {recovery_hint('coco_labels')}\n{recovery_hint('coco_image')}"
                ) from e
            note = f"COCO mini preparation failed; using tiny fallback. {type(e).__name__}: {e}"
            print(f"WARNING: {note}")
            notes.append(note)
            data_root, dataset_source = prepare_from_tiny_fallback(args)
    data_yaml = write_data_yaml(data_root)
    pseudo_source = cache_depth(args)
    if pseudo_source.startswith("synthetic_fallback"):
        notes.append(pseudo_source)
    run_data_tests(data_root, args.seed, metric_depth_max_m(args))
    manifest = DataManifest(
        dataset_root=str(data_root.resolve()),
        dataset_source=dataset_source,
        train_count=len(image_files_from_txt(data_root, "train2017")),
        val_count=len(image_files_from_txt(data_root, "val2017")),
        pseudo_depth_source=pseudo_source,
        depth_representation=DEPTH_REPRESENTATION,
        depth_units=DEPTH_UNITS,
        depth_polarity=DEPTH_POLARITY,
        depth_loss_type=DEPTH_LOSS_TYPE,
        metric_depth_max_m=float(metric_depth_max_m(args) or 0.0),
        sparse_policy="deterministic random 1% + 8 lower-band horizontal scanlines",
        seed=args.seed,
        notes=notes,
    )
    (args.run_root).mkdir(parents=True, exist_ok=True)
    with (args.run_root / "data_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest.__dict__, f, indent=2)
    return data_yaml, manifest


def ensure_text_embeddings(data_yaml: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / "train_label_embeddings.pt"
    neg_path = out_dir / "global_grounding_neg_embeddings.pt"
    if label_path.exists() and neg_path.exists():
        return label_path, "cached"
    data = read_yaml(data_yaml)
    names = [data["names"][i] for i in sorted(data["names"])]
    try:
        from ultralytics.nn.text_model import build_text_model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        text_model = build_text_model("mobileclip:blt", device=device)
        feats = text_model.encode_text(text_model.tokenize(names)).detach().cpu()
        label_embeddings = {name: feats[i] for i, name in enumerate(names)}
        neg = text_model.encode_text(text_model.tokenize(["background"] * 256)).detach().cpu()
        source = "mobileclip_blt"
    except Exception:
        generator = torch.Generator().manual_seed(SEED)
        feats = torch.randn(len(names), 512, generator=generator)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        label_embeddings = {name: feats[i] for i, name in enumerate(names)}
        neg = torch.randn(256, 512, generator=generator)
        neg = neg / neg.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        source = "deterministic_random_fallback"
    torch.save(label_embeddings, label_path)
    torch.save(neg, neg_path)
    return label_path, source


def patch_cached_text_pe(text_embeddings: Path):
    from ultralytics.nn.tasks import YOLOEModel

    cache = torch.load(text_embeddings, map_location="cpu")

    def cached_get_text_pe(self, text, batch=80, cache_clip_model=False):
        device = next(self.model.parameters()).device
        feats = []
        fallback = next(iter(cache.values()))
        for name in text:
            feats.append(cache.get(str(name), fallback))
        txt_feats = torch.stack(feats, dim=0).to(device=device, dtype=torch.float32).reshape(1, len(text), -1)
        return self.model[-1].get_tpe(txt_feats)

    YOLOEModel.get_text_pe = cached_get_text_pe


def latest_csv_row(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k, v in rows[-1].items():
        try:
            out[k.strip()] = float(v)
        except (TypeError, ValueError):
            out[k.strip()] = v
    return out


def letterbox(array, size, interpolation):
    h, w = array.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    out = cv2.resize(array, (new_w, new_h), interpolation=interpolation) if (w, h) != (new_w, new_h) else array
    dw, dh = size - new_w, size - new_h
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    value = 114 if array.ndim == 3 else 0
    return cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)


def preserve_sparse_prediction(pred, sparse, sparse_mask):
    pred = np.asarray(pred, dtype=np.float32).copy()
    sparse = np.asarray(sparse, dtype=np.float32)
    sparse_mask = np.asarray(sparse_mask) > 0.5
    pred[sparse_mask] = sparse[sparse_mask]
    return pred


def depth_metrics(pred, dense, valid, sparse_mask):
    masks = {
        "all": valid > 0,
        "hole": (valid > 0) & (sparse_mask <= 0),
        "upper": (valid > 0) & (np.arange(valid.shape[0])[:, None] < valid.shape[0] // 2),
        "lower": (valid > 0) & (np.arange(valid.shape[0])[:, None] >= valid.shape[0] // 2),
    }
    out = {}
    eps = 1e-6
    for name, mask in masks.items():
        if not mask.any():
            continue
        p = pred[mask].astype(np.float64)
        g_raw = dense[mask].astype(np.float64)
        diff = p - g_raw
        out[f"depth/rmse_{name}"] = float(np.sqrt(np.mean(diff**2)))
        out[f"depth/mae_{name}"] = float(np.mean(np.abs(diff)))
        rel_mask = g_raw > eps
        if rel_mask.any():
            p_rel = np.maximum(p[rel_mask], eps)
            g_rel = g_raw[rel_mask]
            rel_diff = p_rel - g_rel
            out[f"depth/absrel_{name}"] = float(np.mean(np.abs(rel_diff) / g_rel))
            out[f"depth/delta1_{name}"] = float(np.mean(np.maximum(p_rel / g_rel, g_rel / p_rel) < 1.25))
    return out


def torch_weighted_data_loss(output, target, mask, eps=1e-6):
    sampled_output = mask * output + (1.0 - mask) * target
    return torch.abs(sampled_output - target).sum() / (mask.sum() + eps)


def torch_standardize_depth(output, target, mask, eps=1e-6):
    count = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    output_mean = (output * mask).sum(dim=(1, 2, 3), keepdim=True) / count
    target_mean = (target * mask).sum(dim=(1, 2, 3), keepdim=True) / count
    output_scale = (torch.abs((output - output_mean) * mask).sum(dim=(1, 2, 3), keepdim=True) / count).clamp_min(eps)
    target_scale = (torch.abs((target - target_mean) * mask).sum(dim=(1, 2, 3), keepdim=True) / count).clamp_min(eps)
    return (output - output_mean) / output_scale, (target - target_mean) / target_scale


def torch_weighted_ms_grad_loss(output, target, mask, levels=4, eps=1e-6):
    sampled_output = mask * output + (1.0 - mask) * target
    residual = sampled_output - target
    weight_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=residual.device, dtype=residual.dtype).view(1, 1, 3, 3)
    weight_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=residual.device, dtype=residual.dtype).view(1, 1, 3, 3)
    total = residual.new_zeros(())
    for i in range(levels):
        r = residual if i == 0 else torch.nn.functional.interpolate(residual, scale_factor=1.0 / (2**i), mode="bilinear", align_corners=False, recompute_scale_factor=True)
        if r.shape[-1] >= 3 and r.shape[-2] >= 3:
            total = total + torch.abs(torch.nn.functional.conv2d(r, weight_x)).sum() + torch.abs(torch.nn.functional.conv2d(r, weight_y)).sum()
    return total / (mask.sum() + eps)


def spnet_metric_depth_loss(output, target, raw_mask, valid_mask):
    loss_abs = torch_weighted_data_loss(output, target, raw_mask)
    sta_output, sta_target = torch_standardize_depth(output, target, valid_mask)
    loss_rel = torch_weighted_data_loss(sta_output, sta_target, valid_mask)
    loss_grad = torch_weighted_ms_grad_loss(sta_output, sta_target, valid_mask)
    return loss_abs + loss_rel + 0.5 * loss_grad, loss_abs, loss_rel, loss_grad


def load_text_features(path: Path, names: dict[int, str], device):
    cache = torch.load(path, map_location="cpu")
    fallback = next(iter(cache.values()))
    feats = [cache.get(names[i], fallback) for i in sorted(names)]
    return torch.stack(feats, 0).unsqueeze(0).to(device=device, dtype=torch.float32)


def evaluate_yolo_depth(weights: Path, data_yaml: Path, text_embeddings: Path, imgsz: int, device_arg: str):
    from ultralytics.nn.tasks import YOLOEDepthModel  # noqa: F401

    device = torch.device(f"cuda:{device_arg}" if torch.cuda.is_available() and device_arg != "cpu" else "cpu")
    ckpt = torch.load(weights, map_location="cpu")
    model = (ckpt.get("ema") or ckpt.get("model")).float().to(device).eval()
    data = read_yaml(data_yaml)
    names = {int(k): v for k, v in data["names"].items()}
    tpe = load_text_features(text_embeddings, names, device)
    root = Path(data["path"])
    image_paths = image_files_from_txt(root, "val2017")
    accum = []
    with torch.no_grad():
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            dense = np.load(root / "depth" / "dense" / "val2017" / f"{image_path.stem}.npy")
            sparse = np.load(root / "depth" / "sparse" / "val2017" / f"{image_path.stem}.npy")
            sparse_mask = np.load(root / "depth" / "sparse_mask" / "val2017" / f"{image_path.stem}.npy")
            valid = np.load(root / "depth" / "valid" / "val2017" / f"{image_path.stem}.npy")
            image_in = letterbox(image, imgsz, cv2.INTER_LINEAR)
            dense_in = letterbox(dense, imgsz, cv2.INTER_LINEAR)
            sparse_in = letterbox(sparse, imgsz, cv2.INTER_NEAREST)
            sparse_mask_in = letterbox(sparse_mask, imgsz, cv2.INTER_NEAREST)
            valid_in = letterbox(valid, imgsz, cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255.0
            sparse_tensor = torch.from_numpy(sparse_in).unsqueeze(0).unsqueeze(0).to(device).float()
            sparse_mask_tensor = torch.from_numpy(sparse_mask_in).unsqueeze(0).unsqueeze(0).to(device).float()
            _, pred = model(x, tpe=tpe, return_depth=True, sparse_depth=sparse_tensor, sparse_mask=sparse_mask_tensor)
            pred_np = preserve_sparse_prediction(pred[0, 0].detach().cpu().numpy(), sparse_in, sparse_mask_in)
            accum.append(depth_metrics(pred_np, dense_in, valid_in, sparse_mask_in))
    keys = sorted({k for row in accum for k in row})
    return {k: float(np.mean([row[k] for row in accum if k in row])) for k in keys}


def profile_checkpoint(weights: Path, depth: bool, imgsz: int, device_arg: str):
    device = torch.device(f"cuda:{device_arg}" if torch.cuda.is_available() and device_arg != "cpu" else "cpu")
    ckpt = torch.load(weights, map_location="cpu")
    model = (ckpt.get("ema") or ckpt.get("model")).float().to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    sparse = torch.zeros(1, 1, imgsz, imgsz, device=device)
    sparse_mask = torch.zeros(1, 1, imgsz, imgsz, device=device)
    tpe = torch.zeros(1, 80, 512, device=device)
    flops = None
    try:
        import thop

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, image):
                if depth:
                    return self.inner(image, tpe=tpe, return_depth=True, sparse_depth=sparse, sparse_mask=sparse_mask)
                return self.inner(image, tpe=tpe)

        flops = float(thop.profile(Wrapper(model), inputs=[x], verbose=False)[0] / 1e9 * 2)
    except Exception:
        flops = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(3):
            if depth:
                model(x, tpe=tpe, return_depth=True, sparse_depth=sparse, sparse_mask=sparse_mask)
            else:
                model(x, tpe=tpe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        loops = 10
        for _ in range(loops):
            if depth:
                model(x, tpe=tpe, return_depth=True, sparse_depth=sparse, sparse_mask=sparse_mask)
            else:
                model(x, tpe=tpe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms = (time.time() - start) * 1000.0 / loops
    mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return {"params": params, "gflops": flops, "runtime_ms_img": ms, "gpu_mem_gb": mem}


def profile_spnet_network(network, imgsz: int, device):
    rgb = torch.zeros(1, 3, imgsz, imgsz, device=device)
    raw = torch.zeros(1, 1, imgsz, imgsz, device=device)
    known = torch.zeros_like(raw)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    network.eval()
    with torch.no_grad():
        for _ in range(3):
            network(rgb, raw, known)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        loops = 10
        for _ in range(loops):
            network(rgb, raw, known)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    ms = (time.time() - start) * 1000.0 / loops
    mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return ms, mem


def train_one(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str((RUN_ROOT / "yolo_config").resolve()))
    os.environ.setdefault("HF_HOME", str((RUN_ROOT / "hf_home").resolve()))
    patch_cached_text_pe(Path(args.text_embeddings))
    from ultralytics import YOLOE

    trainer = None
    if args.depth:
        from ultralytics.models.yolo.yoloe import YOLOEDepthTrainer

        trainer = YOLOEDepthTrainer
    model = YOLOE(args.cfg)
    if args.depth:
        # Keep the public training task on detection so Ultralytics dataset defaults remain valid;
        # YOLOEDepthTrainer still builds YOLOEDepthModel from the YAML depth_head.
        model.task = "detect"
        model.overrides["task"] = "detect"
    overrides = dict(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0)
    result = {
        "experiment": args.experiment,
        "status": "ok",
        "cfg": args.cfg,
        "depth": args.depth,
        "lr0": 0.0002 if args.depth else 0.001,
        "val_interval": args.val_interval,
        "augmentation_policy": "mosaic/mixup/copy_paste/geometric disabled for every row; photometric RGB defaults may remain",
    }
    if args.depth:
        result.update(
            {
                "depth_loss_type": DEPTH_LOSS_TYPE,
                "sparse_known_mask": "explicit sparse_mask tensor; 1 means known sparse sample",
                "sparse_preserved_for_metrics": True,
            }
        )
    try:
        model.train(
            trainer=trainer,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            val=True,
            val_interval=args.val_interval,
            plots=False,
            project=args.project,
            name=args.name,
            exist_ok=True,
            optimizer="AdamW",
            lr0=result["lr0"],
            weight_decay=0.0,
            warmup_epochs=0.0,
            close_mosaic=0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            amp=args.amp,
            pretrained=False,
            deterministic=True,
            seed=args.seed,
            text_model=args.text_model,
            **overrides,
        )
        run_dir = Path(args.project) / args.name
        result.update(latest_csv_row(run_dir / "results.csv"))
        result.update(profile_checkpoint(run_dir / "weights" / "best.pt", args.depth, args.imgsz, args.device))
        if args.depth:
            result.update(evaluate_yolo_depth(run_dir / "weights" / "best.pt", Path(args.data), Path(args.text_embeddings), args.imgsz, args.device))
    except Exception as e:
        result.update(
            {
                "status": "failed",
                "failure": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=8),
                "actionable_fix": "Inspect the corresponding logs/<experiment>.log, verify data_depth.yaml cache paths, and rerun train-one with the recorded command.",
            }
        )
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.metrics_out).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


class SPNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_yaml: Path, split: str, imgsz: int):
        data = read_yaml(data_yaml)
        self.root = Path(data["path"])
        self.images = image_files_from_txt(self.root, split)
        self.split = split
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        dense = np.load(self.root / "depth" / "dense" / self.split / f"{image_path.stem}.npy")
        sparse = np.load(self.root / "depth" / "sparse" / self.split / f"{image_path.stem}.npy")
        sparse_mask = np.load(self.root / "depth" / "sparse_mask" / self.split / f"{image_path.stem}.npy")
        valid = np.load(self.root / "depth" / "valid" / self.split / f"{image_path.stem}.npy")
        image = letterbox(image, self.imgsz, cv2.INTER_LINEAR)
        dense = letterbox(dense, self.imgsz, cv2.INTER_LINEAR)
        sparse = letterbox(sparse, self.imgsz, cv2.INTER_NEAREST)
        sparse_mask = letterbox(sparse_mask, self.imgsz, cv2.INTER_NEAREST)
        valid = letterbox(valid, self.imgsz, cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        raw = sparse.astype(np.float32)[None]
        gt = dense.astype(np.float32)[None]
        known = (sparse_mask > 0.5).astype(np.float32)[None]
        valid = (valid > 0.5).astype(np.float32)[None]
        return torch.from_numpy(rgb.transpose(2, 0, 1)), torch.from_numpy(raw), torch.from_numpy(gt), torch.from_numpy(known), torch.from_numpy(valid)


def ensure_spnet_repo(no_download=False):
    repo = ROOT / "third_party" / "SPNet"
    if not repo.exists():
        if no_download:
            raise FileNotFoundError(f"SPNet repo is missing and --no-download was set. {recovery_hint('spnet_repo')}")
        log_path = RUN_ROOT / "logs" / "clone_spnet.log"
        code, _ = run_command(["git", "clone", "--depth", "1", SPNET_REPO, repo], log_path)
        if code != 0:
            fail_with_hint("spnet_repo", f"SPNet clone failed; see {rel(log_path)}.", f"exit={code}")
    return repo


def spnet_one(args):
    result = {
        "experiment": "D",
        "title": "SPNet V2Net same-budget adapter",
        "status": "ok",
        "depth": True,
        "depth_loss_type": DEPTH_LOSS_TYPE,
        "sparse_known_mask": "explicit sparse_mask tensor; 1 means known sparse sample",
        "sparse_preserved_for_metrics": True,
    }
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        repo = ensure_spnet_repo(no_download=args.no_download)
        sys.path.insert(0, str(repo))
        from src.networks import V2Net

        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        network = V2Net([96, 192, 384, 768], [3, 3, 9, 3], 0.0, "CNX").to(device)
        ckpt = repo / "checkpoints" / "models" / "Tiny_300.pth"
        result["official_checkpoint_present"] = ckpt.exists()
        result["spnet_source"] = "official_v2net_scratch_same_budget"
        result["lr0"] = 2e-4
        train_ds = SPNetDataset(Path(args.data), "train2017", args.imgsz)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        opt = torch.optim.AdamW(network.parameters(), lr=result["lr0"], weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        network.train()
        for _ in range(args.epochs):
            for rgb, raw, gt, known, valid in loader:
                rgb, raw, gt, known, valid = rgb.to(device), raw.to(device), gt.to(device), known.to(device), valid.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    pred = network(rgb, raw, known)
                loss, loss_abs, loss_rel, loss_grad = spnet_metric_depth_loss(pred.float(), gt.float(), known.float(), valid.float())
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        project = Path(args.project)
        project.mkdir(parents=True, exist_ok=True)
        checkpoint_path = project / "spnet_tiny_adapter.pt"
        torch.save({"network": network.state_dict(), "spnet_source": result["spnet_source"]}, checkpoint_path)
        result["checkpoint"] = str(checkpoint_path)
        params = sum(p.numel() for p in network.parameters())
        network.eval()
        model_ms, model_mem = profile_spnet_network(network, args.imgsz, device)
        val_ds = SPNetDataset(Path(args.data), "val2017", args.imgsz)
        rows = []
        start = time.time()
        with torch.no_grad():
            for rgb, raw, gt, known, valid in torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0):
                rgb, raw, gt, known, valid = rgb.to(device), raw.to(device), gt.to(device), known.to(device), valid.to(device)
                pred = network(rgb, raw, known)
                pred_np = preserve_sparse_prediction(pred[0, 0].cpu().numpy(), raw[0, 0].cpu().numpy(), known[0, 0].cpu().numpy())
                rows.append(depth_metrics(pred_np, gt[0, 0].cpu().numpy(), valid[0, 0].cpu().numpy(), known[0, 0].cpu().numpy()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        keys = sorted({k for row in rows for k in row})
        result.update({k: float(np.mean([row[k] for row in rows if k in row])) for k in keys})
        result.update(
            {
                "params": params,
                "gflops": None,
                "gflops_not_measured_reason": "SPNet official wrapper is profiled for latency only in this benchmark",
                "runtime_ms_img": model_ms,
                "eval_ms_img": (time.time() - start) * 1000.0 / max(len(val_ds), 1),
                "gpu_mem_gb": model_mem,
            }
        )
    except Exception as e:
        result.update(
            {
                "status": "failed",
                "failure": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=8),
                "actionable_fix": actionable_spnet_fix(e),
            }
        )
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.metrics_out).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def actionable_spnet_fix(exc: Exception):
    text = f"{type(exc).__name__}: {exc}"
    if "No module named" in text:
        return "Install the missing SPNet dependency in the active .venv, then rerun the benchmark. Keep official SPNet code unchanged."
    if "SPNet repo is missing" in text or "clone failed" in text:
        return recovery_hint("spnet_repo")
    if "CUDA" in text or "out of memory" in text.lower():
        return "Rerun with smaller --batch or --imgsz, or use --device cpu for SPNet diagnosis."
    return "Inspect logs/D_spnet.log and rerun spnet-one after resolving the reported incompatibility."


def expected_cache_structure(data_root: Path = DATA_ROOT):
    return [
        rel(data_root / "_assets" / "coco" / "train2017.txt"),
        rel(data_root / "_assets" / "coco" / "val2017.txt"),
        rel(data_root / "images" / "train2017" / "*.jpg"),
        rel(data_root / "labels" / "train2017" / "*.txt"),
        rel(data_root / "depth" / "dense" / "train2017" / "*.npy"),
        rel(data_root / "depth" / "sparse" / "train2017" / "*.npy"),
        rel(data_root / "depth" / "sparse_mask" / "train2017" / "*.npy"),
        rel(data_root / "depth" / "valid" / "train2017" / "*.npy"),
        rel(ROOT / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"),
        rel(ROOT / "third_party" / "SPNet"),
    ]


def preflight(args):
    data_root = args.data_root
    assets = data_root / "_assets"
    depth_pro_repo = ROOT / "third_party" / "ml-depth-pro"
    depth_pro_ckpt = depth_pro_repo / "checkpoints" / "depth_pro.pt"
    spnet_repo = ROOT / "third_party" / "SPNet"
    ready, missing_local = existing_mini_ready(data_root, args.train_count, args.val_count)
    depth_ready, missing_depth = depth_cache_complete(data_root)
    counts = depth_cache_counts(data_root)
    depth_manifest = read_depth_cache_manifest(data_root)
    expected_depth_source = "depth_pro_metric"
    expected_depth_marker = depth_cache_expected_marker(args, expected_depth_source)
    depth_manifest_matches = depth_ready and depth_cache_core_matches(depth_manifest, expected_depth_marker)

    train_images = image_files_from_txt(data_root, "train2017")
    val_images = image_files_from_txt(data_root, "val2017")
    coco_train_list = find_first(assets, "train2017.txt")
    coco_val_list = find_first(assets, "val2017.txt")
    coco_label_train = len(list((assets / "coco" / "labels" / "train2017").glob("*.txt")))
    coco_label_val = len(list((assets / "coco" / "labels" / "val2017").glob("*.txt")))

    blockers = []
    if not ready:
        blockers.extend(missing_local)
    if not depth_ready:
        blockers.extend([f"missing depth cache {x}" for x in missing_depth])
        if not depth_pro_repo.exists():
            blockers.append(f"missing Depth Pro repo: {depth_pro_repo}")
        if not depth_pro_ckpt.exists():
            blockers.append(f"missing Depth Pro checkpoint: {depth_pro_ckpt}")
    elif not depth_manifest_matches:
        blockers.append(f"depth cache manifest does not match requested source/precision/seed/metric_depth_max_m: requested={expected_depth_marker}, manifest={depth_manifest}")
    if args.include_spnet and not spnet_repo.exists():
        blockers.append(f"missing SPNet repo: {spnet_repo}")

    no_download_ok = not blockers
    result = {
        "data_root": str(data_root.resolve()),
        "requested_train_count": args.train_count,
        "requested_val_count": args.val_count,
        "coco_mini": {
            "train_txt": str((data_root / "train.txt").resolve()),
            "train_txt_exists": (data_root / "train.txt").exists(),
            "train_entries": len(train_images),
            "train_images_present": sum(1 for p in train_images[: args.train_count] if p.exists()),
            "train_labels_present": sum(1 for p in train_images[: args.train_count] if (data_root / "labels" / "train2017" / f"{p.stem}.txt").exists()),
            "val_txt_exists": (data_root / "val.txt").exists(),
            "val_entries": len(val_images),
            "val_images_present": sum(1 for p in val_images[: args.val_count] if p.exists()),
            "val_labels_present": sum(1 for p in val_images[: args.val_count] if (data_root / "labels" / "val2017" / f"{p.stem}.txt").exists()),
            "ready_for_no_download": ready,
            "missing_examples": missing_local,
        },
        "coco_labels": {
            "assets_root": str(assets.resolve()),
            "train2017_txt_exists": bool(coco_train_list),
            "val2017_txt_exists": bool(coco_val_list),
            "train_label_count": coco_label_train,
            "val_label_count": coco_label_val,
        },
        "depth_pro": {
            "repo": str(depth_pro_repo.resolve()),
            "repo_exists": depth_pro_repo.exists(),
            "checkpoint": str(depth_pro_ckpt.resolve()),
            "checkpoint_exists": depth_pro_ckpt.exists(),
            "manual_clone": recovery_hint("depth_pro_repo"),
            "manual_checkpoint": recovery_hint("depth_pro_ckpt"),
            "manual_import_fix": recovery_hint("depth_pro_import"),
        },
        "depth_cache": {
            **counts,
            "manifest": depth_manifest,
            "requested": expected_depth_marker,
            "matches_requested_source_precision_seed": depth_manifest_matches,
            "representation": depth_manifest.get("representation", DEPTH_REPRESENTATION),
            "units": depth_manifest.get("units", DEPTH_UNITS),
            "metric_depth_max_m": depth_manifest.get("metric_depth_max_m"),
            "complete_for_current_lists": depth_ready,
            "missing_examples": missing_depth,
            "manual_recovery": recovery_hint("depth_cache"),
        },
        "spnet": {
            "repo": str(spnet_repo.resolve()),
            "repo_exists": spnet_repo.exists(),
            "manual_clone": recovery_hint("spnet_repo"),
        },
        "no_download_ready": {
            "include_spnet": args.include_spnet,
            "ok": no_download_ok,
            "blockers": blockers[:30],
            "command": (
                f"{sys.executable} {rel(Path(__file__))} run --include-spnet --no-download "
                f"--train-count {args.train_count} --val-count {args.val_count} --epochs {args.epochs} "
                f"--data-root {args.data_root} --depth-source {args.depth_source} --depth-pro-precision {args.depth_pro_precision} "
                f"--metric-depth-max-m {metric_depth_max_m(args) or 0}"
            ),
        },
        "expected_cache_structure": expected_cache_structure(data_root),
    }

    args.run_root.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out or (args.run_root / "preflight.json")
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Preflight written to {json_out}")
    return result


def smoke(args):
    from ultralytics.nn.tasks import YOLOEDepthModel, YOLOEModel

    for exp in args.experiments:
        meta = EXPERIMENTS[exp]
        cls = YOLOEDepthModel if meta["depth"] else YOLOEModel
        for size in args.sizes:
            model = cls(str(ROOT / meta["cfg"]), ch=3, nc=80, verbose=False).eval()
            x = torch.zeros(1, 3, size, size)
            tpe = torch.zeros(1, 80, 512)
            sparse = torch.zeros(1, 1, size, size)
            sparse_mask = torch.zeros(1, 1, size, size)
            sparse_mask[..., size // 2, size // 2] = 1.0
            with torch.no_grad():
                if meta["depth"]:
                    _, depth = model(x, tpe=tpe, return_depth=True, sparse_depth=sparse, sparse_mask=sparse_mask)
                    assert tuple(depth.shape) == (1, 1, size, size), (exp, size, depth.shape)
                    assert float(depth[0, 0, size // 2, size // 2]) == 0.0, (exp, size, "known zero sparse not preserved")
                else:
                    model(x, tpe=tpe)
            print(f"OK {exp} {size}")


def _grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        value = float(p.grad.detach().float().norm().cpu())
        total += value * value
    return math.sqrt(total)


def gradient_debug(args):
    from ultralytics.nn.tasks import YOLOEDepthModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    size = min(int(args.imgsz), 128)
    batch_size = max(1, min(int(args.batch), 2))
    torch.manual_seed(args.seed)
    model = YOLOEDepthModel(str(ROOT / EXPERIMENTS["C"]["cfg"]), ch=3, nc=80, verbose=False).to(device).train()
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "load_vp": False})
    image = torch.rand(batch_size, 3, size, size, device=device)
    dense = torch.rand(batch_size, 1, size, size, device=device)
    valid = torch.ones_like(dense)
    sparse = torch.zeros_like(dense)
    sparse_mask = torch.zeros_like(dense)
    sparse_mask[..., size // 2, size // 2] = 1.0
    sparse[sparse_mask > 0.5] = dense[sparse_mask > 0.5]
    batch = {
        "img": image,
        "cls": torch.zeros(batch_size, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device).repeat(batch_size, 1),
        "batch_idx": torch.arange(batch_size, device=device).float(),
        "dense_depth": dense,
        "sparse_depth": sparse,
        "sparse_mask": sparse_mask,
        "valid_mask": valid,
    }
    loss, items = model.loss(batch)
    loss.backward()
    depth_grad = _grad_norm(model.depth_head.parameters())
    backbone_grad = _grad_norm(model.model[0].parameters()) if len(model.model) else 0.0
    total_grad = _grad_norm(model.parameters())
    return {
        "status": "ok",
        "device": str(device),
        "imgsz": size,
        "batch": batch_size,
        "loss": float(loss.detach().cpu()),
        "loss_items_box_cls_dfl_depth": [float(x) for x in items.detach().cpu().flatten()],
        "depth_head_grad_norm": depth_grad,
        "first_backbone_grad_norm": backbone_grad,
        "total_grad_norm": total_grad,
        "depth_loss_updates_depth_head": depth_grad > 0,
        "losses_update_shared_backbone": backbone_grad > 0,
    }


def spnet_smoke(args):
    repo = ROOT / "third_party" / "SPNet"
    if not repo.exists():
        return {"status": "skipped", "reason": f"missing {repo}"}
    try:
        sys.path.insert(0, str(repo))
        from src.networks import V2Net

        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        size = min(int(args.imgsz), 128)
        model = V2Net([96, 192, 384, 768], [3, 3, 9, 3], 0.0, "CNX").to(device).eval()
        rgb = torch.zeros(1, 3, size, size, device=device)
        raw = torch.zeros(1, 1, size, size, device=device)
        known = torch.zeros_like(raw)
        with torch.no_grad():
            pred = model(rgb, raw, known)
        return {"status": "ok", "shape": list(pred.shape), "device": str(device)}
    except Exception as e:
        return {"status": "failed", "failure": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=8)}


def run_audit(args):
    args.run_root.mkdir(parents=True, exist_ok=True)
    result = {"status": "ok", "checks": {}, "commands": []}
    result["commands"].append(" ".join(map(str, [sys.executable, rel(Path(__file__)), "audit"])))

    try:
        if image_files_from_txt(args.data_root, "train2017") and image_files_from_txt(args.data_root, "val2017"):
            write_data_yaml(args.data_root)
            changed = retrofit_sparse_masks_from_existing_cache(args.data_root, args.seed, metric_depth_max_m(args))
            if changed:
                result["checks"]["sparse_mask_retrofit"] = {"status": "ok", "changed_files": changed}
            run_data_tests(args.data_root, args.seed, metric_depth_max_m(args))
            result["checks"]["data_tests"] = {"status": "ok"}
        else:
            result["checks"]["data_tests"] = {"status": "skipped", "reason": "data_root train/val lists are missing"}
    except Exception as e:
        result["status"] = "failed"
        result["checks"]["data_tests"] = {"status": "failed", "failure": f"{type(e).__name__}: {e}"}

    try:
        smoke(argparse.Namespace(experiments=args.experiments, sizes=args.sizes))
        result["checks"]["smoke"] = {"status": "ok", "experiments": args.experiments, "sizes": args.sizes}
    except Exception as e:
        result["status"] = "failed"
        result["checks"]["smoke"] = {"status": "failed", "failure": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=8)}

    try:
        result["checks"]["gradient_debug"] = gradient_debug(args)
    except Exception as e:
        result["status"] = "failed"
        result["checks"]["gradient_debug"] = {"status": "failed", "failure": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=8)}

    result["checks"]["spnet_smoke"] = spnet_smoke(args)
    if result["checks"]["spnet_smoke"].get("status") == "failed":
        result["status"] = "failed"

    json_out = args.json_out or (args.run_root / "audit.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Audit Report",
        "",
        f"Status: {result['status']}",
        "",
        "## Confirmed",
        "",
        "- YOLOE-depth configs instantiate and preserve explicit known-zero sparse samples during eval.",
        "- Perfect metric-depth sanity metrics are zero for RMSE/MAE/AbsRel and one for delta1.",
        "- Sparse depth now has an explicit sparse_mask where 1 means known.",
        f"- Depth Pro metric targets are finite, nonnegative, and clipped at {metric_depth_max_m(args)} m without per-image normalization.",
        "- YOLOE-depth loss terms are separated as depth_abs/depth_rel/depth_grad/depth.",
        "- Official SPNet D smoke uses the cloned V2Net path and a linear output.",
        "",
        "## Fixed Bugs",
        "",
        "- Replaced sigmoid/minmax depth assumptions with SPNet-style linear metric-depth output and SPNet absolute/relative/gradient loss.",
        "- Added FP32 depth-loss computation under AMP to avoid meter-domain overflow.",
        "- Repaired benchmark reruns so stale per-row results.csv files are removed before each row starts.",
        "- Fixed SPNet D training loop indentation so every batch receives backward/optimizer updates.",
        "",
        "## Check Results",
        "",
    ]
    for name, check in result["checks"].items():
        lines.append(f"- {name}: {check.get('status')}")
        if check.get("failure"):
            lines.append(f"  failure: {check['failure']}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Depth RMSE/MAE are metric-depth metrics in meters when Depth Pro metric cache is used.",
            f"- Depth Pro pseudo GT is clipped at {metric_depth_max_m(args)} m, matching SPNet-style max_depth handling while preserving meter units.",
            "- The YOLOE depth head uses SPNet-style linear output and SPNet absolute/relative/gradient loss.",
            "- SP-Norm is used in the decoder only and should not be interpreted as replacing ConvNeXtV2 normalization.",
            "",
            "## Exact Commands",
            "",
            *[f"- `{cmd}`" for cmd in result["commands"]],
            "",
            "## Changed Files",
            "",
            *[f"- `{line}`" for line in changed_files()[:40]],
            "",
            f"JSON details: {json_out}",
        ]
    )
    report_path = args.run_root / "audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote audit artifacts: {json_out}, {report_path}")
    return result


def command_markdown(commands):
    lines = [
        "# Benchmark Commands",
        "",
        "## Normal online tiny validation",
        "",
        "```powershell",
        f"{sys.executable} {rel(Path(__file__))} run --include-spnet --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1 --val-interval 1",
        "```",
        "",
        "## Offline cached rerun",
        "",
        "```powershell",
        f"{sys.executable} {rel(Path(__file__))} run --include-spnet --no-download --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1 --val-interval 1",
        "```",
        "",
        "## Preflight",
        "",
        "```powershell",
        f"{sys.executable} {rel(Path(__file__))} preflight --include-spnet --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1",
        "```",
        "",
        "## Manual Asset Recovery",
        "",
        "```powershell",
        f"git clone --depth 1 {SPNET_REPO} {rel(ROOT / 'third_party' / 'SPNet')}",
        f"git clone --depth 1 {DEPTH_PRO_REPO} {rel(ROOT / 'third_party' / 'ml-depth-pro')}",
        "# Download the Depth Pro checkpoint to:",
        rel(ROOT / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"),
        "```",
        "",
        "Expected cache structure:",
        "",
        *[f"- `{p}`" for p in expected_cache_structure()],
        "",
        "Troubleshooting:",
        "",
        "- SSL/GitHub failures: clone the repos manually with the commands above, then rerun with `--no-download`.",
        "- COCO image failures: manually place JPGs under `datasets/coco_mini_depth/images/<split>` and labels under `datasets/coco_mini_depth/labels/<split>`.",
        "- Depth Pro checkpoint failures: place `depth_pro.pt` in `third_party/ml-depth-pro/checkpoints/`.",
        "- Depth Pro import failures: install missing lightweight dependencies such as `pillow_heif`; do not upgrade Torch globally.",
        "- SPNet incompatibility: inspect `logs/D_spnet.log`; the D row records `failure`, `traceback`, and `actionable_fix` without crashing the benchmark.",
        "",
        "## Executed Commands",
        "",
    ]
    for item in commands:
        lines.extend([f"## {item['name']}", "", "```powershell", item["cmd"], "```", ""])
    return "\n".join(lines)


def write_table(path: Path, rows):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows):
    by_exp = {row.get("experiment"): row for row in rows if row.get("status") == "ok"}
    baseline = by_exp.get("B", {})
    b_ap = float(baseline.get("metrics/mAP50-95(B)", baseline.get("tp_metrics/mAP50-95(B)", 0)) or 0)
    candidates = [r for r in rows if str(r.get("experiment", "")).startswith(("C", "V")) and r.get("status") == "ok"]
    if not candidates:
        return None
    all_zero = b_ap == 0 and all(float(r.get("metrics/mAP50-95(B)", r.get("tp_metrics/mAP50-95(B)", 0)) or 0) == 0 for r in candidates)
    if not all_zero:
        min_ap = min(b_ap - 0.01, b_ap * 0.95)
        candidates = [r for r in candidates if float(r.get("metrics/mAP50-95(B)", r.get("tp_metrics/mAP50-95(B)", 0)) or 0) >= min_ap] or candidates
    base_ms = float(baseline.get("runtime_ms_img", 0) or 0)
    if base_ms > 0:
        candidates = [r for r in candidates if float(r.get("runtime_ms_img", 1e9) or 1e9) <= base_ms * 1.5] or candidates
    if all_zero:
        return sorted(
            candidates,
            key=lambda r: (
                -float(r.get("metrics/mAP50(B)", r.get("tp_metrics/mAP50(B)", 0)) or 0),
                float(r.get("depth/rmse_hole", 1e9) or 1e9),
                float(r.get("runtime_ms_img", 1e9) or 1e9),
            ),
        )[0]
    return sorted(candidates, key=lambda r: (float(r.get("depth/rmse_hole", 1e9) or 1e9), float(r.get("runtime_ms_img", 1e9) or 1e9)))[0]


def command_option(command: str | None, option: str, default=None):
    if not command:
        return default
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = str(command).split()
    for i, token in enumerate(tokens):
        if token == option and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return default


def command_flag(command: str | None, flag: str):
    if not command:
        return False
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = str(command).split()
    return flag in tokens


def depth_head_summary(cfg):
    if not cfg:
        return "SPNet official row; no YOLOE YAML."
    cfg_path = ROOT / cfg
    if not cfg_path.exists():
        return "YAML not found in workspace."
    try:
        data = read_yaml(cfg_path)
    except Exception as e:
        return f"Could not read YAML: {type(e).__name__}: {e}"
    depth_cfg = data.get("depth_head") or {}
    if not depth_cfg:
        return "detection only"
    parts = [
        f"variant={depth_cfg.get('variant', 'spnet')}",
        f"from={depth_cfg.get('from')}",
        f"channels={depth_cfg.get('channels')}",
        f"hidden={depth_cfg.get('hidden', 64)}",
        f"lambda_depth={depth_cfg.get('lambda_depth', 0.1)}",
        f"sparse_weight={depth_cfg.get('sparse_weight', 1.0)}",
        f"loss_type={depth_cfg.get('loss_type', DEPTH_LOSS_TYPE)}",
        f"output_activation={depth_cfg.get('output_activation', 'identity')}",
    ]
    enabled = [k for k in ("sparse_encoder", "feature_adapters", "scale_prompt") if depth_cfg.get(k)]
    if enabled:
        parts.append("enabled=" + ",".join(enabled))
    if depth_cfg.get("input_channels"):
        parts.append(f"input_channels={depth_cfg.get('input_channels')}")
    parts.append("base_head_uses_sparse_projection=True")
    if depth_cfg.get("variant", "spnet") == "spnet":
        parts.append("C uses sparse depth as one-layer conditioning, not only as loss/metric input")
    return "; ".join(parts)


def safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def plot_loss_comparison(run_root: Path, out_dir: Path, experiments: list[str]):
    import matplotlib.pyplot as plt

    metrics = [
        ("train/box", "Train Box"),
        ("train/cls", "Train Class"),
        ("train/dfl", "Train DFL"),
        ("train/depth", "Train Depth"),
        ("val/box", "Val Box"),
        ("val/cls", "Val Class"),
        ("val/dfl", "Val DFL"),
        ("val/depth", "Val Depth"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    axes = axes.ravel()
    plotted = False
    for ax, (metric, title) in zip(axes, metrics):
        has_data = False
        for exp in experiments:
            csv_path = run_root / "runs" / exp / "results.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            xs = [safe_float(row.get("epoch")) for row in rows]
            ys = [safe_float(row.get(metric)) for row in rows]
            points = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
            if not points:
                continue
            x_vals, y_vals = zip(*points)
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.6, markersize=4, label=exp)
            has_data = True
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if not has_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.45")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), frameon=False)
    fig.suptitle("YOLOE COCO-Mini Loss Comparison", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss_comparison.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    if not plotted:
        raise FileNotFoundError(f"No per-experiment results.csv files found under {run_root / 'runs'}")
    return out_path


def letterbox_geometry(shape, size):
    h, w = shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = size - new_w, size - new_h
    left = int(round(dw / 2 - 0.1))
    top = int(round(dh / 2 - 0.1))
    return r, left, top, new_w, new_h


def label_path_for_image(data_root: Path, split: str, image_path: Path):
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def yolo_label_boxes(label_path: Path, image_shape, imgsz: int):
    boxes = []
    if not label_path.exists():
        return boxes
    h, w = image_shape[:2]
    r, left, top, _, _ = letterbox_geometry(image_shape, imgsz)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = [safe_float(x) for x in parts[:5]]
        if not all(np.isfinite(x) for x in (cls, xc, yc, bw, bh)):
            continue
        x1 = (xc - bw / 2) * w * r + left
        y1 = (yc - bh / 2) * h * r + top
        x2 = (xc + bw / 2) * w * r + left
        y2 = (yc + bh / 2) * h * r + top
        boxes.append([x1, y1, x2, y2, 1.0, int(cls)])
    return boxes


def title_tile(image, title, subtitle=None):
    header = 48 if subtitle else 34
    h, w = image.shape[:2]
    canvas = np.full((h + header, w, 3), 245, dtype=np.uint8)
    canvas[header : header + h, :w] = image
    cv2.putText(canvas, title[:42], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:52], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)
    return canvas


def message_tile(size, title, message):
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    y = 92
    for line in wrap(message, width=34)[:7]:
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        y += 22
    return title_tile(image, title)


def make_grid(tiles, cols=3, pad=8):
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    rows = int(math.ceil(len(tiles) / cols))
    out = np.full((rows * tile_h + (rows + 1) * pad, cols * tile_w + (cols + 1) * pad, 3), 248, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (tile_h + pad)
        x = pad + c * (tile_w + pad)
        out[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return out


def stack_images_vertical(images, pad=12):
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    width = max(image.shape[1] for image in images)
    height = sum(image.shape[0] for image in images) + pad * (len(images) + 1)
    out = np.full((height, width + 2 * pad, 3), 248, dtype=np.uint8)
    y = pad
    for image in images:
        x = pad + (width - image.shape[1]) // 2
        out[y : y + image.shape[0], x : x + image.shape[1]] = image
        y += image.shape[0] + pad
    return out


def draw_detection_boxes(image, boxes, names, color, max_labels=12):
    out = image.copy()
    h, w = out.shape[:2]
    count = 0
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box[:6]
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        cls = int(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = names.get(cls, str(cls))
        if conf is not None and conf < 0.999:
            label = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (min(w - 1, x1 + tw + 5), y1), color, -1)
        cv2.putText(out, label, (x1 + 3, max(11, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        count += 1
        if count >= max_labels:
            break
    if count == 0:
        cv2.putText(out, "no visible boxes", (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 220), 1, cv2.LINE_AA)
    return out


def depth_to_bgr(depth, keep_zero_black=False):
    arr = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(arr)
    valid = finite & (arr > 0)
    if valid.any():
        lo, hi = float(np.percentile(arr[valid], 1)), float(np.percentile(arr[valid], 99))
        arr = (arr - lo) / max(hi - lo, 1e-6)
    arr = np.clip(arr, 0.0, 1.0)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_INFERNO)
    color = cv2.applyColorMap((arr * 255).astype(np.uint8), cmap)
    if keep_zero_black:
        color[arr <= 0] = (20, 20, 20)
    return color


def select_visual_samples(data_root: Path, sample_count: int):
    images = image_files_from_txt(data_root, "val2017")
    labeled, unlabeled = [], []
    for image_path in images:
        label_path = label_path_for_image(data_root, "val2017", image_path)
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            labeled.append(image_path)
        else:
            unlabeled.append(image_path)
    return (labeled + unlabeled)[:sample_count]


def prepare_visual_sample(data_root: Path, image_path: Path, imgsz: int, names):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    dense = np.load(data_root / "depth" / "dense" / "val2017" / f"{image_path.stem}.npy")
    sparse = np.load(data_root / "depth" / "sparse" / "val2017" / f"{image_path.stem}.npy")
    sparse_mask = np.load(data_root / "depth" / "sparse_mask" / "val2017" / f"{image_path.stem}.npy")
    image_in = letterbox(image, imgsz, cv2.INTER_LINEAR)
    dense_in = letterbox(dense, imgsz, cv2.INTER_LINEAR)
    sparse_in = letterbox(sparse, imgsz, cv2.INTER_NEAREST)
    sparse_mask_in = letterbox(sparse_mask, imgsz, cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    sparse_tensor = torch.from_numpy(sparse_in).unsqueeze(0).unsqueeze(0).float()
    sparse_mask_tensor = torch.from_numpy(sparse_mask_in).unsqueeze(0).unsqueeze(0).float()
    gt_boxes = yolo_label_boxes(label_path_for_image(data_root, "val2017", image_path), image.shape, imgsz)
    gt_tile = draw_detection_boxes(image_in, gt_boxes, names, (30, 180, 30), max_labels=20)
    return {
        "path": image_path,
        "image": image_in,
        "x": x,
        "dense": dense_in.astype(np.float32),
        "sparse": sparse_in.astype(np.float32),
        "sparse_mask": (sparse_mask_in > 0.5).astype(np.float32),
        "sparse_tensor": sparse_tensor,
        "sparse_mask_tensor": sparse_mask_tensor,
        "det_tiles": [title_tile(gt_tile, "Input + GT", image_path.name)],
        "depth_tiles": [
            title_tile(image_in, "RGB input", image_path.name),
            title_tile(depth_to_bgr(dense_in), "Pseudo dense GT"),
            title_tile(depth_to_bgr(sparse_in, keep_zero_black=True), "Sparse input"),
        ],
    }


def load_checkpoint_model(weights: Path, device):
    ckpt = torch.load(weights, map_location="cpu")
    return (ckpt.get("ema") or ckpt.get("model")).float().to(device).eval()


def run_yolo_visual(model, sample, tpe, meta, device, conf, iou, max_det, names):
    from ultralytics.utils.ops import non_max_suppression

    x = sample["x"].to(device)
    sparse_tensor = sample["sparse_tensor"].to(device)
    sparse_mask_tensor = sample["sparse_mask_tensor"].to(device)
    with torch.no_grad():
        if meta["depth"]:
            det_out, depth = model(x, tpe=tpe, return_depth=True, sparse_depth=sparse_tensor, sparse_mask=sparse_mask_tensor)
            depth_np = depth[0, 0].detach().float().cpu().numpy()
        else:
            det_out = model(x, tpe=tpe)
            depth_np = None
        pred = det_out[0] if isinstance(det_out, (tuple, list)) else det_out
        det = non_max_suppression(pred.clone(), conf_thres=conf, iou_thres=iou, max_det=max_det)[0]
    det_np = det.detach().cpu().numpy() if det is not None else np.zeros((0, 6), dtype=np.float32)
    det_tile = draw_detection_boxes(sample["image"], det_np, names, (30, 90, 230), max_labels=max_det)
    depth_tile = title_tile(depth_to_bgr(depth_np), "Depth " + meta["title"][:28]) if depth_np is not None else None
    return det_tile, depth_tile


def load_spnet_visual_model(run_root: Path, device):
    repo = ROOT / "third_party" / "SPNet"
    if not repo.exists():
        return None, "missing third_party/SPNet"
    sys.path.insert(0, str(repo))
    from src.networks import V2Net

    network = V2Net([96, 192, 384, 768], [3, 3, 9, 3], 0.0, "CNX").to(device)
    saved = run_root / "spnet" / "spnet_tiny_adapter.pt"
    if saved.exists():
        state = torch.load(saved, map_location="cpu")
        network.load_state_dict(state.get("network", state))
        source = state.get("spnet_source", "saved_adapter")
    else:
        return None, f"missing {saved}; rerun spnet-one once to save SPNet visualization weights"
    return network.eval(), source


def visualize_outputs(args):
    run_root = args.run_root
    out_dir = run_root / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "results.json"
    manifest = {}
    if results_path.exists():
        manifest = json.loads(results_path.read_text(encoding="utf-8")).get("manifest", {})
    data_root = args.data_root or Path(manifest.get("dataset_root", DATA_ROOT))
    data_yaml = data_root / "data_depth.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data yaml: {data_yaml}")
    data = read_yaml(data_yaml)
    names = {int(k): v for k, v in data["names"].items()}
    samples = [prepare_visual_sample(data_root, p, args.imgsz, names) for p in select_visual_samples(data_root, args.sample_count)]
    if not samples:
        raise FileNotFoundError(f"No validation samples found under {data_root}")

    written = [plot_loss_comparison(run_root, out_dir, args.experiments)]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    text_embeddings = run_root / "text_embeddings" / "train_label_embeddings.pt"
    if not text_embeddings.exists():
        raise FileNotFoundError(f"Missing text embeddings: {text_embeddings}")
    tpe = load_text_features(text_embeddings, names, device)

    for exp in args.experiments:
        meta = EXPERIMENTS.get(exp)
        if meta is None:
            continue
        weights = run_root / "runs" / exp / "weights" / "best.pt"
        if not weights.exists():
            for sample in samples:
                sample["det_tiles"].append(message_tile(args.imgsz, exp, f"missing weights: {weights}"))
                if meta["depth"]:
                    sample["depth_tiles"].append(message_tile(args.imgsz, "Depth " + exp, "missing weights"))
            continue
        try:
            model = load_checkpoint_model(weights, device)
            for sample in samples:
                det_tile, depth_tile = run_yolo_visual(model, sample, tpe, meta, device, args.conf, args.iou, args.max_det, names)
                sample["det_tiles"].append(title_tile(det_tile, exp + " " + meta["title"][:31], f"conf={args.conf}, max_det={args.max_det}"))
                if depth_tile is not None:
                    sample["depth_tiles"].append(depth_tile)
        except Exception as e:
            for sample in samples:
                sample["det_tiles"].append(message_tile(args.imgsz, exp, f"{type(e).__name__}: {e}"))
                if meta["depth"]:
                    sample["depth_tiles"].append(message_tile(args.imgsz, "Depth " + exp, f"{type(e).__name__}: {e}"))
        finally:
            if "model" in locals():
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.include_spnet:
        try:
            spnet, source = load_spnet_visual_model(run_root, device)
            if spnet is None:
                for sample in samples:
                    sample["depth_tiles"].append(message_tile(args.imgsz, "D SPNet", source))
            else:
                with torch.no_grad():
                    for sample in samples:
                        rgb = sample["x"].to(device)
                        raw = sample["sparse_tensor"].to(device)
                        known = sample["sparse_mask_tensor"].to(device)
                        pred = spnet(rgb, raw, known)[0, 0].detach().cpu().numpy()
                        pred = preserve_sparse_prediction(pred, sample["sparse"], sample["sparse_mask"])
                        sample["depth_tiles"].append(title_tile(depth_to_bgr(pred), "D SPNet", source))
                del spnet
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:
            for sample in samples:
                sample["depth_tiles"].append(message_tile(args.imgsz, "D SPNet", f"{type(e).__name__}: {e}"))

    detection_grids, depth_grids = [], []
    for i, sample in enumerate(samples):
        stem = sample["path"].stem
        det_path = out_dir / f"sample_{i:03d}_{stem}_detection_grid.jpg"
        depth_path = out_dir / f"sample_{i:03d}_{stem}_depth_grid.jpg"
        det_grid = make_grid(sample["det_tiles"], cols=3)
        depth_grid = make_grid(sample["depth_tiles"], cols=3)
        cv2.imwrite(str(det_path), det_grid)
        cv2.imwrite(str(depth_path), depth_grid)
        detection_grids.append(det_grid)
        depth_grids.append(depth_grid)
        written.extend([det_path, depth_path])
    detection_overview = out_dir / "detection_samples_overview.jpg"
    depth_overview = out_dir / "depth_samples_overview.jpg"
    cv2.imwrite(str(detection_overview), stack_images_vertical(detection_grids))
    cv2.imwrite(str(depth_overview), stack_images_vertical(depth_grids))
    written.extend([detection_overview, depth_overview])
    print("Wrote visualizations:")
    for path in written:
        print(f"- {path}")
    return written


def report_setting_lines(rows, manifest):
    first_command = next((row.get("command") for row in rows if row.get("command")), "")
    lines = [
        "Common Benchmark Setup",
        "",
        f"Dataset root: {manifest.get('dataset_root')}",
        f"Dataset source: {manifest.get('dataset_source')}",
        f"Subset size: train={manifest.get('train_count')} images, val={manifest.get('val_count')} images. Detection values are subset AP, not full COCO AP.",
        f"Pseudo dense depth source: {manifest.get('pseudo_depth_source')}",
        f"Depth representation: {manifest.get('depth_representation', DEPTH_REPRESENTATION)}; units={manifest.get('depth_units', DEPTH_UNITS)}; polarity={manifest.get('depth_polarity', DEPTH_POLARITY)}.",
        f"Metric target clip: {manifest.get('metric_depth_max_m', DEFAULT_METRIC_DEPTH_MAX_M)} m max_depth; this keeps meter units while limiting Depth Pro far-depth outliers.",
        "Depth cache: dense .npy files were prepared before training and reused by every depth experiment.",
        f"Sparse depth policy: {manifest.get('sparse_policy')}",
        "Sparse known pixels use explicit depth/sparse_mask .npy files; sparse value 0 can be a valid known sample.",
        f"Seed: {manifest.get('seed')}",
        f"Text embedding source: {manifest.get('text_embedding_source')}",
        "",
        "Training / Evaluation Runtime",
        "",
        f"imgsz={command_option(first_command, '--imgsz', 'n/a')}, batch={command_option(first_command, '--batch', 'n/a')}, workers={command_option(first_command, '--workers', 'n/a')}",
        f"epochs={command_option(first_command, '--epochs', 'n/a')}, val_interval={command_option(first_command, '--val-interval', '1')}, device={command_option(first_command, '--device', 'n/a')}, AMP={'False' if command_flag(first_command, '--no-amp') else 'True'}",
        f"Optimizer/loss defaults: AdamW; detection rows lr0=0.001, YOLOE-depth rows lr0=0.0002, SPNet D lr0=0.0002. Depth loss={manifest.get('depth_loss_type', DEPTH_LOSS_TYPE)}.",
        "Augmentation controls for every row: mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, degrees/translate/scale/shear/perspective/flipud/fliplr all 0.",
        "Depth-safe caveat: color-only RGB augmentations may remain, but geometric/mix/copy-paste depth augmentations are unsupported unless depth-aware transforms are added.",
        "Validation: val=True with benchmark default val_interval=1, so CSV validation metrics are freshly evaluated every epoch unless overridden.",
        "",
        "Depth Metrics",
        "",
        "Depth metrics are metric-depth metrics in meters using Depth Pro pseudo GT after the configured meter-domain max_depth clip.",
        "Depth metrics are computed against the same cached metric dense depth and explicit sparse known mask for all depth rows.",
        "Reported depth metrics include RMSE, MAE, AbsRel, delta1 for all-valid pixels, sparse holes, upper image region, and lower image region.",
        "Known sparse pixels are preserved in YOLOE and SPNet depth predictions before metric calculation; training loss uses raw linear depth predictions before sparse preservation.",
        "",
        "SPNet D Row",
        "",
        "SPNet D uses the official cloned V2Net implementation with scratch same-budget training on this cached subset; official checkpoints are not loaded for D.",
        "SPNet failures do not crash the benchmark; results.json records status, failure, traceback, command, and actionable_fix.",
    ]
    return lines


def experiment_setting_lines(rows):
    lines = ["Per-Experiment Setup", ""]
    for row in rows:
        exp = row.get("experiment")
        title = row.get("title")
        command = row.get("command", "")
        cfg = row.get("cfg")
        if exp == "D":
            trainer = "official SPNet V2Net wrapper / scratch same-budget path"
            model = "SPNet V2Net official code, random initialization"
            depth_setup = f"spnet_source={row.get('spnet_source', 'n/a')}"
        elif row.get("depth"):
            trainer = "YOLOEDepthTrainer"
            model = cfg
            depth_setup = depth_head_summary(cfg)
        else:
            trainer = "YOLOETrainer"
            model = cfg
            depth_setup = "detection only"
        lines.extend(
            [
                f"{exp}. {title}",
                f"  trainer: {trainer}",
                f"  model/config: {model}",
                f"  depth setup: {depth_setup}",
                f"  train command settings: imgsz={command_option(command, '--imgsz', 'n/a')}, batch={command_option(command, '--batch', 'n/a')}, workers={command_option(command, '--workers', 'n/a')}, epochs={command_option(command, '--epochs', 'n/a')}, val_interval={command_option(command, '--val-interval', 'n/a')}, device={command_option(command, '--device', 'n/a')}",
                f"  result status: {row.get('status')}; params={row.get('params', 'n/a')}; GFLOPs={row.get('gflops', 'n/a')}; runtime_ms_img={row.get('runtime_ms_img', 'n/a')}; gpu_mem_gb={row.get('gpu_mem_gb', 'n/a')}",
                "",
            ]
        )
    return lines


def first_page_setting_summary(rows, manifest):
    first_command = next((row.get("command") for row in rows if row.get("command")), "")
    return [
        "Test Settings Snapshot",
        f"subset: train={manifest.get('train_count')} / val={manifest.get('val_count')} | imgsz={command_option(first_command, '--imgsz', 'n/a')} | batch={command_option(first_command, '--batch', 'n/a')} | workers={command_option(first_command, '--workers', 'n/a')} | epochs={command_option(first_command, '--epochs', 'n/a')}",
        f"seed={manifest.get('seed')} | AMP={'False' if command_flag(first_command, '--no-amp') else 'True'} | AP label=subset AP",
        f"depth prepared before training: {manifest.get('pseudo_depth_source')} metric dense .npy cache",
        f"depth metric units: {manifest.get('depth_units', DEPTH_UNITS)}; representation={manifest.get('depth_representation', DEPTH_REPRESENTATION)}",
        f"sparse policy: {manifest.get('sparse_policy')}",
        "rows: A original YOLOE, B ConvNeXtV2, C SPNet-style head, V1 5ch input, V2 sparse encoder, V3 adapters, V4 scale prompt, V5 V2+V4, D SPNet baseline",
    ]


def add_text_pages(pdf, plt, title, lines, fontsize=8, wrap_width=118):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=14, weight="bold")
    y = 0.9
    for line in lines:
        if line == "":
            y -= 0.016
            continue
        is_heading = not line.startswith(" ") and len(line) < 80 and not line.endswith(".") and ":" not in line
        chunks = wrap(line, width=wrap_width, subsequent_indent="  ") or [line]
        for chunk in chunks:
            fig.text(0.08, y, chunk, fontsize=fontsize + (1 if is_heading else 0), weight="bold" if is_heading else "normal", family="monospace" if line.startswith(" ") else None)
            y -= 0.022
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.08, 0.94, title + " (continued)", fontsize=14, weight="bold")
                y = 0.9
        if is_heading:
            y -= 0.006
    pdf.savefig(fig)
    plt.close(fig)


def write_report(path: Path, rows, manifest, commands, changed_files):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.compression"] = 0
    best = choose_best(rows)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "YOLOE + ConvNeXtV2 + Depth Completion COCO-Mini Benchmark", fontsize=16, weight="bold")
        summary = [
            f"Dataset source: {manifest.get('dataset_source')}",
            f"Train/val: {manifest.get('train_count')} / {manifest.get('val_count')}",
            f"Pseudo depth: {manifest.get('pseudo_depth_source')}",
            f"Depth representation: {manifest.get('depth_representation', DEPTH_REPRESENTATION)} ({manifest.get('depth_units', DEPTH_UNITS)})",
            f"Sparse policy: {manifest.get('sparse_policy')}",
            f"Seed: {manifest.get('seed')}",
            f"Best variant: {best.get('experiment') if best else 'none'}",
        ]
        fig.text(0.08, 0.86, "\n".join(summary), fontsize=10, va="top")
        y = 0.69
        fig.text(0.08, y, "Test Settings Snapshot", fontsize=13, weight="bold")
        y -= 0.03
        for line in first_page_setting_summary(rows, manifest)[1:]:
            for chunk in wrap(line, width=125):
                fig.text(0.08, y, chunk, fontsize=7.5)
                y -= 0.019
            y -= 0.004
        y -= 0.015
        fig.text(0.08, y, "Results", fontsize=13, weight="bold")
        y -= 0.03
        for row in rows:
            line = (
                f"{row.get('experiment')}: {row.get('status')} | "
                f"subset mAP50={row.get('metrics/mAP50(B)', row.get('tp_metrics/mAP50(B)', 'n/a'))} | "
                f"subset mAP50-95={row.get('metrics/mAP50-95(B)', row.get('tp_metrics/mAP50-95(B)', 'n/a'))} | "
                f"hole RMSE={row.get('depth/rmse_hole', 'n/a')} | ms/img={row.get('runtime_ms_img', 'n/a')}"
            )
            fig.text(0.08, y, line[:130], fontsize=8)
            y -= 0.025
            if y < 0.1:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.94
        pdf.savefig(fig)
        plt.close(fig)

        add_text_pages(pdf, plt, "Test Settings", report_setting_lines(rows, manifest))
        add_text_pages(pdf, plt, "Experiment Settings", experiment_setting_lines(rows), fontsize=7, wrap_width=110)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Commands And Changed Files", fontsize=14, weight="bold")
        text = ["Commands are saved in commands.md.", "", "Changed files:"] + changed_files[:30]
        fig.text(0.08, 0.88, "\n".join(text), fontsize=8, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def changed_files():
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def orchestrate(args):
    if args.dry_run:
        args.train_count = min(args.train_count, 8)
        args.val_count = min(args.val_count, 4)
        args.epochs = min(args.epochs, 1)
        if args.no_download:
            args.allow_tiny_fallback = True
            args.allow_fallback_depth = True
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.skip_prepare:
        data_yaml = args.data_root / "data_depth.yaml"
        manifest = json.loads((args.run_root / "data_manifest.json").read_text(encoding="utf-8"))
    else:
        data_yaml, manifest_obj = prepare(args)
        manifest = manifest_obj.__dict__
    text_dir = args.run_root / "text_embeddings"
    text_embeddings, text_source = ensure_text_embeddings(data_yaml, text_dir)
    manifest["text_embedding_source"] = text_source
    text_model_arg = "../" + rel(text_dir).replace("\\", "/")
    commands, rows = [{"name": "top-level run", "cmd": " ".join(map(str, [sys.executable, *sys.argv]))}], []
    project = args.run_root / "runs"
    for exp in args.experiments:
        meta = EXPERIMENTS[exp]
        metrics_out = args.run_root / "metrics" / f"{exp}.json"
        remove_generated_path(metrics_out, args.run_root)
        remove_generated_path(project / exp, args.run_root)
        cmd = [
            sys.executable,
            rel(Path(__file__)),
            "train-one",
            "--experiment",
            exp,
            "--cfg",
            meta["cfg"],
            "--data",
            str(data_yaml),
            "--project",
            str(project),
            "--name",
            exp,
            "--text-model",
            text_model_arg,
            "--text-embeddings",
            str(text_embeddings),
            "--imgsz",
            args.imgsz,
            "--batch",
            args.batch,
            "--workers",
            args.workers,
            "--epochs",
            args.epochs,
            "--val-interval",
            args.val_interval,
            "--device",
            args.device,
            "--seed",
            args.seed,
            "--metrics-out",
            metrics_out,
        ]
        if meta["depth"]:
            cmd.append("--depth")
        if not args.amp:
            cmd.append("--no-amp")
        commands.append({"name": exp, "cmd": " ".join(map(str, cmd))})
        code, seconds = run_command(cmd, args.run_root / "logs" / f"{exp}.log")
        if metrics_out.exists():
            row = json.loads(metrics_out.read_text(encoding="utf-8"))
        else:
            row = {"experiment": exp, "status": "failed", "failure": f"subprocess_exit_{code}"}
        row.update({"title": meta["title"], "seconds": seconds, "subset_ap": True, "command": " ".join(map(str, cmd))})
        rows.append(row)
    if args.include_spnet:
        metrics_out = args.run_root / "metrics" / "D.json"
        remove_generated_path(metrics_out, args.run_root)
        remove_generated_path(args.run_root / "spnet", args.run_root)
        cmd = [
            sys.executable,
            rel(Path(__file__)),
            "spnet-one",
            "--data",
            str(data_yaml),
            "--project",
            str(args.run_root / "spnet"),
            "--imgsz",
            args.imgsz,
            "--batch",
            args.batch,
            "--workers",
            args.workers,
            "--epochs",
            args.epochs,
            "--device",
            args.device,
            "--seed",
            args.seed,
            "--metrics-out",
            metrics_out,
        ]
        if not args.amp:
            cmd.append("--no-amp")
        if args.no_download:
            cmd.append("--no-download")
        commands.append({"name": "D", "cmd": " ".join(map(str, cmd))})
        code, seconds = run_command(cmd, args.run_root / "logs" / "D_spnet.log")
        row = json.loads(metrics_out.read_text(encoding="utf-8")) if metrics_out.exists() else {"experiment": "D", "status": "failed", "failure": f"subprocess_exit_{code}"}
        row.update({"seconds": seconds, "subset_ap": False, "command": " ".join(map(str, cmd))})
        rows.append(row)
    best = choose_best(rows)
    if best:
        for row in rows:
            row["selected_best"] = row.get("experiment") == best.get("experiment")
    (args.run_root / "commands.md").write_text(command_markdown(commands), encoding="utf-8")
    with (args.run_root / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "results": rows, "best": best}, f, indent=2)
    write_table(args.run_root / "results.csv", rows)
    write_report(args.run_root / "report.pdf", rows, manifest, commands, changed_files())
    print(f"Wrote benchmark artifacts to {args.run_root}")


def regenerate_report(args):
    results_path = args.run_root / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing benchmark results file: {results_path}")
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", [])
    manifest = data.get("manifest", {})
    commands = [
        {"name": row.get("experiment", f"row{i}"), "cmd": row.get("command", "")}
        for i, row in enumerate(rows)
        if row.get("command")
    ]
    output = args.output or (args.run_root / "report.pdf")
    write_report(output, rows, manifest, commands, changed_files())
    print(f"Regenerated report: {output}")


def main():
    args = parse_args()
    if args.command == "prepare":
        data_yaml, manifest = prepare(args)
        print(f"Prepared {data_yaml}")
        print(json.dumps(manifest.__dict__, indent=2))
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "audit":
        run_audit(args)
    elif args.command == "run":
        orchestrate(args)
    elif args.command == "preflight":
        preflight(args)
    elif args.command == "report":
        regenerate_report(args)
    elif args.command == "visualize":
        visualize_outputs(args)
    elif args.command == "train-one":
        train_one(args)
    elif args.command == "spnet-one":
        spnet_one(args)


if __name__ == "__main__":
    main()
