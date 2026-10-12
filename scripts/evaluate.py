"""Kvasir ic test ve CVC-ClinicDB dis test uzerinde degerlendirme.

Esikler onceden sabitlendi, testte ayarlanmaz (U-Net 0.5, YOLO guveni 0.25). Cikti:
  results/<tag>/<set>_images.csv   goruntu basina Dice/IoU + tespit sayilari + goruntu ozellikleri
  results/<tag>/<set>_polyps.csv   gercek polip basina: boyut, parlama, yakalandi mi
  results/<tag>/summary.json       ozet + bootstrap %95 guven araliklari

Kullanim:  python scripts/evaluate.py --ckpt checkpoints/unet_r34/best.pt --tag unet_r34
           python scripts/evaluate.py --yolo checkpoints/yolov8s_seg/weights/best.pt --tag yolov8s_seg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polip.data import DATA, cvc_samples, kvasir_subset, read_pair  # noqa: E402
from polip.metrics import (  # noqa: E402
    box_iou, components, detection_summary, dice, iou, match_detections, roc_auc, wilson_ci,
)
from polip.model import UnetPredictor, YoloPredictor  # noqa: E402

SETS = {"kvasir_test": lambda: kvasir_subset("test"), "cvc_external": cvc_samples}


def specular_fraction(img_rgb: np.ndarray, region: np.ndarray | None = None) -> float:
    """Parlama: cok parlak ve renksiz pikseller (HSV V>230, S<40)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    spec = (hsv[..., 2] > 230) & (hsv[..., 1] < 40)
    if region is not None:
        return float(spec[region > 0].mean()) if region.any() else 0.0
    return float(spec.mean())


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_set(predictor, samples, out_dir: Path, name: str) -> dict:
    img_rows, polyp_rows, det_results = [], [], []
    for s in samples:
        img, gt = read_pair(s)
        prob, pred = predictor.predict(img)
        gt_c = components(gt)
        pred_c = components(pred, prob)
        r = match_detections(pred_c, gt_c)
        det_results.append(r)
        n_fp = sum(1 for _, tp in r.scored if not tp)
        img_rows.append({
            "image": s.name, "dice": dice(pred, gt), "iou": iou(pred, gt),
            "n_gt": len(gt_c), "n_gt_hit": r.n_gt_hit, "n_pred": len(pred_c), "n_fp": n_fp,
            "gt_area_frac": float(gt.mean()), "pred_area_frac": float(pred.mean()),
            "brightness": float(img.mean()), "specular_frac": specular_fraction(img),
            "flagged": len(pred_c) > 0, "max_score": max((c.score for c in pred_c), default=0.0),
        })
        # her gercek polip icin: en iyi eslesen tahmin kutusu IoU'su ve yakalanip yakalanmadigi
        for gc in gt_c:
            region = np.zeros_like(gt)
            x0, y0, x1, y1 = gc.box
            region[y0:y1, x0:x1] = gt[y0:y1, x0:x1]
            best = max((box_iou(pc.box, gc.box) for pc in pred_c), default=0.0)
            polyp_rows.append({
                "image": s.name, "area_frac": gc.area / gt.size, "box_w": x1 - x0, "box_h": y1 - y0,
                "best_box_iou": best, "hit": best >= 0.5,
                "mean_prob_inside": float(prob[region > 0].mean()),
                "specular_inside": specular_fraction(img, region),
            })

    for rows, fname in ((img_rows, f"{name}_images.csv"), (polyp_rows, f"{name}_polyps.csv")):
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    d = np.array([r["dice"] for r in img_rows])
    j = np.array([r["iou"] for r in img_rows])
    summ = {
        "n_images": len(img_rows),
        "dice_mean": float(d.mean()), "dice_ci95": bootstrap_ci(d), "dice_median": float(np.median(d)),
        "iou_mean": float(j.mean()), "iou_ci95": bootstrap_ci(j),
        "images_dice_below_0.5": int((d < 0.5).sum()),
        "image_sensitivity": float(np.mean([r["flagged"] for r in img_rows])),
        "image_sensitivity_ci95": wilson_ci(sum(r["flagged"] for r in img_rows), len(img_rows)),
        **detection_summary(det_results),
    }
    return summ


def evaluate_negatives(predictor, paths: list[Path], out_dir: Path) -> tuple[dict, list[float]]:
    """Polipsiz goruntuler: goruntu duzeyi ozgulluk ve yanlis pozitif sayisi."""
    rows = []
    for pth in paths:
        img = cv2.cvtColor(cv2.imread(str(pth)), cv2.COLOR_BGR2RGB)
        prob, pred = predictor.predict(img)
        comps = components(pred, prob)
        rows.append({"image": pth.stem, "n_pred": len(comps), "flagged": len(comps) > 0,
                     "max_score": max((c.score for c in comps), default=0.0),
                     "pred_area_frac": float(pred.mean()), "specular_frac": specular_fraction(img)})
    with open(out_dir / "normal_cecum_images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tn = sum(not r["flagged"] for r in rows)
    summ = {"n_images": len(rows), "specificity": tn / len(rows), "specificity_ci95": wilson_ci(tn, len(rows)),
            "fp_per_image": float(np.mean([r["n_pred"] for r in rows]))}
    return summ, [r["max_score"] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="U-Net kontrol noktasi")
    src.add_argument("--yolo", type=Path, help="YOLOv8-seg agirliklari")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    args = ap.parse_args()

    predictor = UnetPredictor(args.ckpt) if args.ckpt else YoloPredictor(args.yolo)
    out_dir = Path("results") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(args.ckpt or args.yolo), "threshold": predictor.threshold}
    for name in args.sets:
        summary[name] = evaluate_set(predictor, SETS[name](), out_dir, name)
        s = summary[name]
        print(f"{name:13s} n={s['n_images']:4d}  Dice {s['dice_mean']:.3f} [{s['dice_ci95'][0]:.3f}, {s['dice_ci95'][1]:.3f}]"
              f"  IoU {s['iou_mean']:.3f}  sens {s['lesion_sensitivity']:.3f}  prec {s['precision']:.3f}"
              f"  FP/img {s['fp_per_image']:.3f}  AP50 {s['ap50']:.3f}")
    neg_dir = DATA / "normal-cecum"
    if neg_dir.is_dir():
        neg, neg_scores = evaluate_negatives(predictor, sorted(neg_dir.glob("*.jpg")), out_dir)
        summary["normal_cecum"] = neg
        for name in args.sets:
            with open(out_dir / f"{name}_images.csv") as f:
                pos_scores = [float(r["max_score"]) for r in csv.DictReader(f)]
            summary[name]["image_auc_vs_normal_cecum"] = roc_auc(pos_scores, neg_scores)
        lo, hi = neg["specificity_ci95"]
        print(f"normal_cecum  n={neg['n_images']:4d}  ozgulluk {neg['specificity']:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  FP/img {neg['fp_per_image']:.3f}")
        for name in args.sets:
            s = summary[name]
            print(f"{name:13s} goruntu duyarliligi {s['image_sensitivity']:.3f}  AUC(vs normal) {s['image_auc_vs_normal_cecum']:.3f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
