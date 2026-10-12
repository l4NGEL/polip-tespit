"""evaluate.py ciktilarindan hata analizi.

1) Polip boyutuna gore lezyon duyarliligi (ic test vs dis test)
2) Kacirilan polipler: boyut, parlama ve modelin polip icindeki ortalama olasiligi
3) Dis testte Dice'i en dusuk goruntuler icin bindirmeli gorseller (results/<tag>/overlays/)
4) docs/ icin iki sekil: boyuta gore duyarlilik, Dice dagilimi

Kullanim:  python scripts/error_analysis.py --ckpt checkpoints/unet_r34/best.pt --tag unet_r34
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polip.data import cvc_samples, kvasir_subset, read_pair  # noqa: E402
from polip.model import UnetPredictor, YoloPredictor  # noqa: E402

SIZE_BINS = [(0.0, 0.01, "<%1"), (0.01, 0.03, "%1-3"), (0.03, 0.10, "%3-10"), (0.10, 1.01, ">%10")]
NAMES = {"kvasir_test": "Kvasir (ic test)", "cvc_external": "CVC-ClinicDB (dis test)"}


def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def sensitivity_by_size(polyps: list[dict]) -> list[dict]:
    out = []
    for lo, hi, label in SIZE_BINS:
        sel = [p for p in polyps if lo <= p["area_frac"] < hi]
        hits = sum(p["hit"] for p in sel)
        out.append({"bin": label, "n": len(sel), "hit": hits, "sensitivity": hits / len(sel) if sel else float("nan")})
    return out


def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Yesil kontur: gercek polip; kirmizi kontur: tahmin."""
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    for m, color in ((gt, (0, 200, 0)), (pred, (0, 0, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, color, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path)
    src.add_argument("--yolo", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worst", type=int, default=12)
    args = ap.parse_args()
    res = Path("results") / args.tag
    report: dict = {}

    for name in NAMES:
        polyps = read_csv(res / f"{name}_polyps.csv")
        images = read_csv(res / f"{name}_images.csv")
        missed = [p for p in polyps if not p["hit"]]
        hit = [p for p in polyps if p["hit"]]
        report[name] = {
            "sensitivity_by_size": sensitivity_by_size(polyps),
            "missed_n": len(missed),
            "missed_median_area_frac": float(np.median([p["area_frac"] for p in missed])) if missed else None,
            "hit_median_area_frac": float(np.median([p["area_frac"] for p in hit])) if hit else None,
            "missed_median_prob_inside": float(np.median([p["mean_prob_inside"] for p in missed])) if missed else None,
            "missed_partial_box_iou_0.1_0.5": sum(0.1 <= p["best_box_iou"] < 0.5 for p in missed),
            "missed_no_overlap_lt_0.1": sum(p["best_box_iou"] < 0.1 for p in missed),
            "missed_median_specular_inside": float(np.median([p["specular_inside"] for p in missed])) if missed else None,
            "hit_median_specular_inside": float(np.median([p["specular_inside"] for p in hit])) if hit else None,
            "images_with_fp": sum(r["n_fp"] > 0 for r in images),
            "median_gt_area_frac": float(np.median([r["gt_area_frac"] for r in images])),
            "dice_median": float(np.median([r["dice"] for r in images])),
        }

    # en kotu dis test goruntuleri icin bindirmeler
    predictor = UnetPredictor(args.ckpt) if args.ckpt else YoloPredictor(args.yolo)
    ov_dir = res / "overlays"
    ov_dir.mkdir(exist_ok=True)
    images = sorted(read_csv(res / "cvc_external_images.csv"), key=lambda r: r["dice"])[: args.worst]
    lookup = {s.name: s for s in cvc_samples()}
    for r in images:
        s = lookup[str(int(r["image"]))]
        img, gt = read_pair(s)
        _, pred = predictor.predict(img)
        cv2.imwrite(str(ov_dir / f"cvc_{s.name}_dice{r['dice']:.2f}.png"), overlay(img, gt, pred))
    report["worst_cvc_images"] = [{"image": int(r["image"]), "dice": r["dice"], "gt_area_frac": r["gt_area_frac"],
                                   "n_fp": int(r["n_fp"])} for r in images]

    # sekiller
    docs = Path("docs") / args.tag
    docs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.38
    xs = np.arange(len(SIZE_BINS))
    for k, (name, label) in enumerate(NAMES.items()):
        b = report[name]["sensitivity_by_size"]
        vals = [v["sensitivity"] for v in b]
        bars = ax.bar(xs + (k - 0.5) * width, vals, width, label=label, color=["#2f6690", "#d1495b"][k])
        for x, v in zip(bars, b):
            ax.text(x.get_x() + x.get_width() / 2, 0.02, f"n={v['n']}", ha="center", va="bottom", fontsize=7, color="white")
    ax.set_xticks(xs, [b[2] for b in SIZE_BINS])
    ax.set_xlabel("Polip alani / goruntu alani")
    ax.set_ylabel("Lezyon duyarliligi (kutu IoU >= 0.5)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(docs / "sensitivity_by_size.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k, (name, label) in enumerate(NAMES.items()):
        d = [r["dice"] for r in read_csv(res / f"{name}_images.csv")]
        ax.hist(d, bins=np.linspace(0, 1, 21), alpha=0.6, density=True, label=label, color=["#2f6690", "#d1495b"][k])
    ax.set_xlabel("Goruntu basina Dice")
    ax.set_ylabel("Yogunluk")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(docs / "dice_distribution.png", dpi=150)

    (res / "error_analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "worst_cvc_images"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
