# YOLOE + ConvNeXtV2 + SPNet-Style Metric Depth Benchmark

This benchmark compares YOLOE detection baselines and YOLOE depth-completion
variants on a fixed COCO-mini subset. Depth GT is generated with Apple Depth Pro
and kept in metric units.

All detection AP values are subset AP, not full COCO AP.

## Current Depth Representation

Depth Pro outputs metric depth. The benchmark stores that output as meter-valued
pseudo GT and applies an explicit SPNet-style max-depth clip to suppress
far-depth outliers that otherwise destabilize meter-domain training:

```text
depth representation: metric_depth
depth units: meters
depth source: depth_pro_metric
default max_depth: 100.0 m
```

Depth RMSE/MAE in `results.csv`, `results.json`, and `report.pdf` are therefore
meter RMSE/MAE against Depth Pro pseudo GT after the configured meter clip. This
is not per-image min/max normalization; values remain in meters.

Depth Anything V2 support was removed from the benchmark path to avoid mixing
relative depth and metric depth in the same workflow.

## Rows

- A: YOLOE-v8s original backbone, detection only
- B: YOLOE + ConvNeXtV2-Small, detection only
- C: YOLOE + ConvNeXtV2-Small + SPNet-style metric depth head
- V1: 5-channel shared backbone, RGB + sparse metric depth + sparse known mask
- V2: RGB ConvNeXtV2 + shallow sparse-depth encoder
- V3: depth-path C2/C3/C4/C5 adapters
- V4: sparse-stat scale prompt with FiLM-style gating
- V5: V2 + V4
- D: official SPNet `V2Net` wrapper, scratch same-budget training

## SPNet-Style Depth Logic

YOLOE depth heads now follow the SPNet-style output/loss convention:

- Final depth output is linear/identity, not sigmoid-normalized.
- Known sparse metric pixels are preserved only for evaluation/visualization.
- Training loss uses raw depth predictions before sparse preservation.
- Depth loss is:

```text
absolute sparse L1
+ robust-standardized dense L1
+ 0.5 * robust-standardized multi-scale Sobel gradient loss
```

The detection loss remains the original YOLOE detection loss. For depth rows:

```text
total loss = YOLOE detection loss + lambda_depth * SPNet-style depth loss
```

Logged depth terms:

```text
train/depth_abs
train/depth_rel
train/depth_grad
train/depth
val/depth_abs
val/depth_rel
val/depth_grad
val/depth
```

## Sparse Depth Cache

Depth rows use four cache trees:

```text
depth/dense/<split>/*.npy        # metric dense Depth Pro pseudo GT, meters
depth/sparse/<split>/*.npy       # metric sparse depth, meters
depth/sparse_mask/<split>/*.npy  # 1 means known sparse sample
depth/valid/<split>/*.npy        # 1 means valid dense GT pixel
```

`sparse_mask` is separate from sparse values because metric depth can validly be
zero after invalid-value sanitization.

Sparse policy:

```text
deterministic random 1% valid pixels + 8 lower-band horizontal scanlines
seed = 20260525
```

## Prepare Metric Depth Cache

Regenerate metric Depth Pro cache. Use this after changing from the older
normalized cache or after changing `--metric-depth-max-m`:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py prepare --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --force-depth-cache --no-download --train-count 512 --val-count 128 --epochs 2 --data-root datasets\coco_mini_depth_fullrun --run-root runs\coco_mini_depth_benchmark_fullrun
```

Without `--force-depth-cache`, the script validates the manifest and refuses to
reuse a cache whose source/precision/seed/representation/units/max-depth do not
match. Existing metric Depth Pro caches can be repaired in place by rerunning
`prepare` without `--force-depth-cache`; dense/sparse arrays are clipped to the
configured meter limit and sparse masks are regenerated deterministically.

## Preflight

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py preflight --include-spnet --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 512 --val-count 128 --epochs 2 --data-root datasets\coco_mini_depth_fullrun --run-root runs\coco_mini_depth_benchmark_fullrun --json-out runs\coco_mini_depth_benchmark_fullrun\preflight_metric_depth_pro.json
```

Preflight checks:

- COCO mini images/labels
- Depth Pro repo/checkpoint
- metric dense/sparse/sparse_mask/valid cache counts
- cache manifest source/precision/seed/representation/units/max-depth
- SPNet repo
- whether `--no-download` can run

## Audit

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py audit --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 512 --val-count 128 --epochs 2 --data-root datasets\coco_mini_depth_fullrun --run-root runs\coco_mini_depth_benchmark_audit --sizes 128 320
```

Audit outputs:

```text
runs/coco_mini_depth_benchmark_audit/audit.json
runs/coco_mini_depth_benchmark_audit/audit_report.md
```

## Dry Benchmark

For a tiny workflow check:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py run --include-spnet --skip-prepare --no-download --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 8 --val-count 4 --epochs 1 --val-interval 1 --data-root datasets\coco_mini_depth_audit --run-root runs\coco_mini_depth_benchmark_audit --imgsz 320 --batch 2 --workers 0
```

## Full Benchmark

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py run --include-spnet --no-download --depth-source depth-pro --depth-pro-precision fp16 --metric-depth-max-m 100 --train-count 512 --val-count 128 --epochs 2 --val-interval 1 --data-root datasets\coco_mini_depth_fullrun --run-root runs\coco_mini_depth_benchmark_depthpro_metric --imgsz 320 --batch 2 --workers 0
```

Keep `--epochs` identical for every row when changing the training budget.
`--val-interval` defaults to `1`, so validation metrics in `results.csv` are
freshly evaluated every epoch. Increase it only when you intentionally want less
frequent validation.
Detection-only rows use `lr0=0.001`; YOLOE-depth and SPNet D rows use
`lr0=0.0002` for stable SPNet-style meter-depth training.

## Report And Visualizations

Regenerate report only:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py report --run-root runs\coco_mini_depth_benchmark_depthpro_metric
```

Generate loss and prediction grids:

```powershell
.\.venv\Scripts\python.exe .\tools\coco_mini_depth_benchmark.py visualize --run-root runs\coco_mini_depth_benchmark_depthpro_metric --data-root datasets\coco_mini_depth_fullrun --sample-count 3 --imgsz 320 --device 0 --conf 0.05 --max-det 8 --include-spnet
```

## Manual Setup

Depth Pro:

```powershell
git clone --depth 1 https://github.com/apple/ml-depth-pro.git third_party\ml-depth-pro
.\.venv\Scripts\python.exe -m pip install pillow_heif
```

Place the Depth Pro checkpoint here:

```text
third_party/ml-depth-pro/checkpoints/depth_pro.pt
```

SPNet:

```powershell
git clone --depth 1 https://github.com/Wang-xjtu/SPNet.git third_party\SPNet
```

Do not upgrade the main `.venv` Torch for Depth Pro or SPNet.

## Expected Cache Layout

```text
datasets/coco_mini_depth_fullrun/train.txt
datasets/coco_mini_depth_fullrun/val.txt
datasets/coco_mini_depth_fullrun/data_depth.yaml
datasets/coco_mini_depth_fullrun/depth/dense/train2017/*.npy
datasets/coco_mini_depth_fullrun/depth/dense/val2017/*.npy
datasets/coco_mini_depth_fullrun/depth/sparse/train2017/*.npy
datasets/coco_mini_depth_fullrun/depth/sparse/val2017/*.npy
datasets/coco_mini_depth_fullrun/depth/sparse_mask/train2017/*.npy
datasets/coco_mini_depth_fullrun/depth/sparse_mask/val2017/*.npy
datasets/coco_mini_depth_fullrun/depth/valid/train2017/*.npy
datasets/coco_mini_depth_fullrun/depth/valid/val2017/*.npy
datasets/coco_mini_depth_fullrun/depth/source/train2017/*.json
datasets/coco_mini_depth_fullrun/depth/source/val2017/*.json
datasets/coco_mini_depth_fullrun/depth/cache_manifest.json
```

## Troubleshooting

- If Depth Pro clone/download fails, clone manually and place `depth_pro.pt` at the path above.
- If the cache manifest says `per_image_minmax_normalized`, regenerate with `--force-depth-cache`.
- If `--no-download` fails, run `preflight` and follow the exact blocker paths.
- If SPNet cannot run, D records `status`, `failure`, `traceback`, `command`, and `actionable_fix` without crashing the benchmark.
- Do not enable geometric/mix/copy-paste depth augmentation unless depth-aware transforms are implemented.
