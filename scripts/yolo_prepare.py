"""Kvasir-SEG train/val bolumunu YOLOv8-seg formatina cevirir (data/yolo/).

Ayni splits/kvasir_split.json kullanilir; Kvasir test ve CVC-ClinicDB buraya hic girmez.
Her maske bileseni bir ornek (polygon) olur; MIN_COMPONENT_PX'ten kucuk lekeler atilir.

Kullanim:  python scripts/yolo_prepare.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polip.data import DATA, kvasir_subset, read_pair  # noqa: E402
from polip.metrics import MIN_COMPONENT_PX  # noqa: E402

OUT = DATA / "yolo"


def mask_to_polygons(m: np.ndarray) -> list[np.ndarray]:
    h, w = m.shape
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < MIN_COMPONENT_PX or len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float32) / (w, h)
        polys.append(pts)
    return polys


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for part in ("train", "val"):
        (OUT / "images" / part).mkdir(parents=True)
        (OUT / "labels" / part).mkdir(parents=True)
        n_inst = 0
        for s in kvasir_subset(part):
            _, m = read_pair(s)
            polys = mask_to_polygons(m)
            n_inst += len(polys)
            shutil.copy(s.image, OUT / "images" / part / s.image.name)
            lines = ["0 " + " ".join(f"{v:.6f}" for v in p.ravel()) for p in polys]
            (OUT / "labels" / part / f"{s.name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{part}: {len(kvasir_subset(part))} goruntu, {n_inst} polip ornegi")
    (OUT / "polyp.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
