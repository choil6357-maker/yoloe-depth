#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Sample training parameters. Edit this block when you want a different run.
export TRAINER="${TRAINER:-depth}"
export MODEL_CFG="${MODEL_CFG:-ultralytics/cfg/models/v8/yoloe_convnextv2_spnet_sample.yaml}"
export DATA="${DATA:-tiny_video100/data_depth.yaml}"
export EPOCHS="${EPOCHS:-8}"
export IMGSZ="${IMGSZ:-320}"
export BATCH="${BATCH:-2}"
export DEVICE="${DEVICE:-0}"
export WORKERS="${WORKERS:-0}"
export PROJECT="${PROJECT:-runs}"
export NAME="${NAME:-yoloe_convnextv2_spnet_sample_320_guided_gn}"
export LR0="${LR0:-0.001}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.0}"
export AMP="${AMP:-False}"
export DEBUG_VIDEO="${DEBUG_VIDEO:-True}"
export DEBUG_VIDEO_FPS="${DEBUG_VIDEO_FPS:-10}"
export DEBUG_VIDEO_MAX_FRAMES="${DEBUG_VIDEO_MAX_FRAMES:-100}"
export DEBUG_VIDEO_PATH="${DEBUG_VIDEO_PATH:-$PROJECT/$NAME/debug_depth_prediction.mp4}"
export DEBUG_VIDEO_CONF="${DEBUG_VIDEO_CONF:-0.05}"
export DEBUG_VIDEO_IOU="${DEBUG_VIDEO_IOU:-0.7}"
export DEBUG_VIDEO_MAX_DET="${DEBUG_VIDEO_MAX_DET:-20}"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT_DIR/runs/yolo_config}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/runs/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$YOLO_CONFIG_DIR" "$HF_HOME" runs/yoloe_mobileclip_blt_text

if [[ ! -f "$MODEL_CFG" ]]; then
  echo "Model config not found: $MODEL_CFG" >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "Data config not found: $DATA" >&2
  exit 1
fi

if [[ "$TRAINER" == "depth" || "$TRAINER" == "spnet" ]]; then
  if ! compgen -G "tiny_video100/depth/dense/train/*.npy" > /dev/null; then
    echo "Precomputed dense depth files are missing under tiny_video100/depth/dense/train." >&2
    echo "Run: $PY prepare_yoloe_depth_sample.py --data tiny_video100/data_depth.yaml --split train" >&2
    exit 1
  fi
  if ! compgen -G "tiny_video100/depth/sparse/train/*.npy" > /dev/null; then
    echo "Precomputed sparse depth files are missing under tiny_video100/depth/sparse/train." >&2
    echo "Run: $PY prepare_yoloe_depth_sample.py --data tiny_video100/data_depth.yaml --split train" >&2
    exit 1
  fi
  if ! compgen -G "tiny_video100/depth/valid/train/*.npy" > /dev/null; then
    echo "Precomputed valid-mask files are missing under tiny_video100/depth/valid/train." >&2
    echo "Run: $PY prepare_yoloe_depth_sample.py --data tiny_video100/data_depth.yaml --split train" >&2
    exit 1
  fi
fi

"$PY" - <<'PY'
from pathlib import Path

import torch
from ultralytics.nn.text_model import build_text_model

out_dir = Path("runs/yoloe_mobileclip_blt_text")
out_dir.mkdir(parents=True, exist_ok=True)

label_path = out_dir / "train_label_embeddings.pt"
neg_path = out_dir / "global_grounding_neg_embeddings.pt"

if not label_path.exists() or not neg_path.exists():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_model = build_text_model("mobileclip:blt", device=device)

    person = text_model.encode_text(text_model.tokenize(["person"]))[0].cpu()
    neg = text_model.encode_text(text_model.tokenize(["background"])).cpu()

    torch.save({"person": person}, label_path)
    torch.save(neg, neg_path)
    print(f"Wrote MobileCLIP-BLT sample text embeddings to {out_dir}")
PY

"$PY" - <<'PY'
import os

import torch

from ultralytics import YOLOE
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.nn.text_model import build_text_model

try:
    from tiny_video100.run_train_100 import cache_labels_no_pool
    from ultralytics.data import dataset as yolo_dataset

    yolo_dataset.YOLODataset.cache_labels = cache_labels_no_pool
except Exception as e:
    print(f"Cache label patch skipped: {e}")


def mobileclip_blt_get_text_pe(self, text, batch=80, cache_clip_model=False):
    device = next(self.model.parameters()).device
    text_model = build_text_model("mobileclip:blt", device=device)
    text_token = text_model.tokenize(text)
    txt_feats = text_model.encode_text(text_token)
    txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
    return self.model[-1].get_tpe(txt_feats)


# Training uses MobileCLIP-BLT cached embeddings; validation builds PE from mobileclip_blt.pt directly.
YOLOEModel.get_text_pe = mobileclip_blt_get_text_pe

trainer = None
trainer_name = os.getenv("TRAINER", "").lower()
if trainer_name in {"depth", "spnet"}:
    from ultralytics.models.yolo.yoloe import YOLOEDepthTrainer

    trainer = YOLOEDepthTrainer

is_depth_train = trainer is not None
data_path = os.environ["DATA"]
depth_aug_overrides = (
    dict(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0)
    if is_depth_train
    else {}
)

model = YOLOE(os.environ["MODEL_CFG"])
model.train(
    trainer=trainer,
    data=data_path,
    epochs=int(os.environ["EPOCHS"]),
    imgsz=int(os.environ["IMGSZ"]),
    batch=int(os.environ["BATCH"]),
    device=os.environ["DEVICE"],
    workers=int(os.environ["WORKERS"]),
    val=False,
    plots=False,
    project=os.environ["PROJECT"],
    name=os.environ["NAME"],
    exist_ok=True,
    optimizer="AdamW",
    lr0=float(os.environ["LR0"]),
    weight_decay=float(os.environ["WEIGHT_DECAY"]),
    warmup_epochs=float(os.environ["WARMUP_EPOCHS"]),
    close_mosaic=0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    amp=os.getenv("AMP", "False").lower() == "true",
    pretrained=False,
    deterministic=False,
    text_model="../runs/yoloe_mobileclip_blt_text",
    **depth_aug_overrides,
)
PY

if [[ "$TRAINER" == "depth" || "$TRAINER" == "spnet" ]]; then
  case "${DEBUG_VIDEO,,}" in
    true|1|yes|y)
      "$PY" save_yoloe_depth_debug_video.py \
        --weights "$PROJECT/$NAME/weights/best.pt" \
        --data "$DATA" \
        --imgsz "$IMGSZ" \
        --output "$DEBUG_VIDEO_PATH" \
        --fps "$DEBUG_VIDEO_FPS" \
        --max-frames "$DEBUG_VIDEO_MAX_FRAMES" \
        --device "$DEVICE" \
        --conf "$DEBUG_VIDEO_CONF" \
        --iou "$DEBUG_VIDEO_IOU" \
        --max-det "$DEBUG_VIDEO_MAX_DET"
      ;;
  esac
fi
