"""Precompute sample dense/sparse depth tensors for the YOLOE depth sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="tiny_video100/data_depth.yaml", help="Dataset yaml with depth output dirs.")
    parser.add_argument("--split", default="train", help="Dataset split to prepare.")
    return parser.parse_args()


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def split_value(value, split):
    return value.get(split) or value.get("train") if isinstance(value, dict) else value


def make_depth(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    h, w = gray.shape
    ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    dense = np.clip(0.65 * gray + 0.35 * ramp, 0, 1).astype(np.float32)

    step = max(min(h, w) // 16, 1)
    sparse = np.zeros_like(dense, dtype=np.float32)
    sparse[::step, ::step] = dense[::step, ::step]
    valid = np.ones_like(dense, dtype=np.float32)
    return dense, sparse, valid


def main():
    args = parse_args()
    data_path = Path(args.data)
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_path.parent))
    image_root = resolve_path(root, data[args.split])
    dense_root = resolve_path(root, split_value(data["dense_depth"], args.split))
    sparse_root = resolve_path(root, split_value(data["sparse_depth"], args.split))
    valid_root = resolve_path(root, split_value(data["valid_mask"], args.split))
    for directory in (dense_root, sparse_root, valid_root):
        directory.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not image_files:
        raise FileNotFoundError(f"No images found under {image_root}")

    for image_file in image_files:
        image = cv2.imread(str(image_file))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_file}")
        dense, sparse, valid = make_depth(image)
        np.save(dense_root / f"{image_file.stem}.npy", dense, allow_pickle=False)
        np.save(sparse_root / f"{image_file.stem}.npy", sparse, allow_pickle=False)
        np.save(valid_root / f"{image_file.stem}.npy", valid, allow_pickle=False)

    print(f"Prepared {len(image_files)} precomputed depth samples")
    print(f"dense_depth={dense_root}")
    print(f"sparse_depth={sparse_root}")
    print(f"valid_mask={valid_root}")


if __name__ == "__main__":
    main()
