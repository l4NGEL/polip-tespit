"""YOLOv8-seg egitimi (COCO on-egitimli agirliktan), Kvasir train/val ile.

Kullanim:  python scripts/train_yolo.py --model yolov8s-seg.pt --epochs 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=352)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolov8s_seg")
    args = ap.parse_args()

    # Ortamda MLflow kuruluysa Ultralytics onu kendiliginden acar; MLflow'un dosya deposu artik
    # hata firlatiyor. Kalici ayari degistirmeden yalnizca bu surecte kapat.
    import ultralytics.utils.callbacks.mlflow as _mlflow_cb

    _mlflow_cb.callbacks.clear()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "data" / "yolo" / "polyp.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=0, deterministic=True,
        project=str(ROOT / "checkpoints"), name=args.name, exist_ok=True,
        flipud=0.5, fliplr=0.5, degrees=20, mosaic=1.0, workers=4, verbose=False, plots=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
