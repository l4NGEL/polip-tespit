"""Segmentasyon ve lezyon duzeyi tespit metrikleri.

Segmentasyon: goruntu basina Dice ve IoU.
Tespit: tahmin maskesinin baglantili bilesenleri birer "tespit" kabul edilir, skoru
bilesen icindeki ortalama olasiliktir. Bir tespit, kutu IoU >= 0.5 ile henuz eslesmemis bir
gercek polipe eslenirse dogru pozitiftir (VOC tarzi, skor sirasina gore acgozlu).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_COMPONENT_PX = 50  # bundan kucuk lekeler (JPEG maske gurultusu, tek piksel tahmin) yok sayilir
EPS = 1e-7


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


@dataclass
class Component:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (dahil degil)
    area: int
    score: float = 1.0


def components(mask: np.ndarray, prob: np.ndarray | None = None) -> list[Component]:
    """Ikili maskenin baglantili bilesenleri; prob verilirse skor bilesen icindeki ortalama olasiliktir."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < MIN_COMPONENT_PX:
            continue
        score = float(prob[labels == i].mean()) if prob is not None else 1.0
        out.append(Component((int(x), int(y), int(x + w), int(y + h)), int(a), score))
    return out


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + EPS)


@dataclass
class DetectionResult:
    """Bir goruntunun tespitleri: (skor, dogru_mu) listesi ve gercek polip sayisi."""

    scored: list[tuple[float, bool]] = field(default_factory=list)
    n_gt: int = 0
    n_gt_hit: int = 0


def match_detections(pred_comps: list[Component], gt_comps: list[Component], iou_thr: float = 0.5) -> DetectionResult:
    res = DetectionResult(n_gt=len(gt_comps))
    used = [False] * len(gt_comps)
    for pc in sorted(pred_comps, key=lambda c: -c.score):
        best, best_j = 0.0, -1
        for j, gc in enumerate(gt_comps):
            if used[j]:
                continue
            v = box_iou(pc.box, gc.box)
            if v > best:
                best, best_j = v, j
        tp = best >= iou_thr
        if tp:
            used[best_j] = True
        res.scored.append((pc.score, tp))
    res.n_gt_hit = sum(used)
    return res


def average_precision(results: list[DetectionResult]) -> float:
    """Tum goruntuler uzerinde AP (VOC 2010+ tum-nokta enterpolasyonu)."""
    scored = sorted((s for r in results for s in r.scored), key=lambda t: -t[0])
    n_gt = sum(r.n_gt for r in results)
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    tps = np.array([t for _, t in scored], dtype=float)
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1.0 - tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, EPS)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_summary(results: list[DetectionResult]) -> dict[str, float]:
    n_gt = sum(r.n_gt for r in results)
    hits = sum(r.n_gt_hit for r in results)
    n_pred = sum(len(r.scored) for r in results)
    tp = sum(t for r in results for _, t in r.scored)
    fp = n_pred - tp
    return {
        "lesion_sensitivity": hits / n_gt if n_gt else float("nan"),
        "precision": tp / n_pred if n_pred else float("nan"),
        "fp_per_image": fp / len(results) if results else float("nan"),
        "ap50": average_precision(results),
        "n_gt_polyps": n_gt,
        "n_images": len(results),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binom oran icin Wilson %95 guven araligi (kucuk n'de normal yaklasimdan guvenilir)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def roc_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Mann-Whitney U ile ROC-AUC: rastgele bir pozitifin rastgele bir negatiften yuksek skor alma olasiligi
    (esitlikler yarim sayilir)."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))
