import argparse
import os
from pathlib import Path
import cv2
from ultralytics import YOLOE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image or video"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yoloe-v8l-seg.pt",
        help="Path or ID of the model checkpoint"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["person"],
        help="List of class names to set for the model"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image (image mode only)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))

    source_suffix = Path(args.source).suffix.lower()
    is_video = source_suffix in {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg"
    }

    if is_video:
        results = model.predict(
            source=args.source,
            device=args.device,
            conf=args.conf,
            save=True,
            verbose=False,
        )
        save_dir = results[0].save_dir if results else "runs/segment/predict"
        print(f"Annotated video saved under: {save_dir}")
        return

    if not args.output:
        base, ext = os.path.splitext(args.source)
        args.output = f"{base}-output{ext}"

    results = model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        save=False,
        verbose=False,
    )
    annotated = results[0].plot()
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to: {args.output}")

if __name__ == "__main__":
    main()
