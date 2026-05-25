# COCO-Mini YOLOE Metric-Depth Benchmark

This benchmark prepares a fixed COCO mini subset, caches Apple Depth Pro
meter-valued pseudo-depth targets, derives deterministic sparse metric depth,
runs YOLOE depth variants, and records an official SPNet row when possible.

Depth Anything V2 is no longer used by this benchmark path.

## Commands

Preflight without downloading:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py preflight --include-spnet --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1
```

Small online validation run:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py run --include-spnet --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1 --val-interval 1
```

Offline rerun after assets are cached:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py run --include-spnet --no-download --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1 --val-interval 1
```

## Metric Depth

Depth Pro outputs metric depth. The cache stores meter values after finite-value
sanitization and a configurable max-depth clip:

```text
representation: metric_depth
units: meters
default max_depth: 100.0 m
```

The YOLOE depth heads use SPNet-style linear output and SPNet-style
absolute/relative/gradient loss. Sparse known pixels are preserved for
evaluation, while training loss uses raw linear predictions.

`--val-interval` defaults to `1`, so YOLO validation metrics are recomputed every
epoch unless you override it.

## Manual Asset Setup

```powershell
git clone --depth 1 https://github.com/Wang-xjtu/SPNet.git third_party/SPNet
git clone --depth 1 https://github.com/apple/ml-depth-pro.git third_party/ml-depth-pro
.\.venv\Scripts\python.exe -m pip install pillow_heif
```

Place the Depth Pro checkpoint here:

```text
third_party/ml-depth-pro/checkpoints/depth_pro.pt
```

Expected cache structure:

```text
datasets/coco_mini_depth/_assets/coco/train2017.txt
datasets/coco_mini_depth/_assets/coco/val2017.txt
datasets/coco_mini_depth/images/train2017/*.jpg
datasets/coco_mini_depth/labels/train2017/*.txt
datasets/coco_mini_depth/depth/dense/train2017/*.npy
datasets/coco_mini_depth/depth/sparse/train2017/*.npy
datasets/coco_mini_depth/depth/sparse_mask/train2017/*.npy
datasets/coco_mini_depth/depth/valid/train2017/*.npy
datasets/coco_mini_depth/depth/cache_manifest.json
third_party/ml-depth-pro/checkpoints/depth_pro.pt
third_party/SPNet
```

## Troubleshooting

- SSL/GitHub failures: clone the repos manually, place `depth_pro.pt` manually, then rerun with `--no-download`.
- COCO download failures: place images under `datasets/coco_mini_depth/images/<split>` and labels under `datasets/coco_mini_depth/labels/<split>`, then rerun `preflight`.
- Metric cache mismatch: rerun `prepare --metric-depth-max-m 100` to repair an existing Depth Pro metric cache, or add `--force-depth-cache` to regenerate it.
- SPNet failures: the D row records `failure`, `traceback`, and `actionable_fix` in `results.json`; the rest of the benchmark continues.
