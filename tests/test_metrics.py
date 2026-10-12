import numpy as np
import pytest

from polip.metrics import (
    Component,
    DetectionResult,
    average_precision,
    box_iou,
    components,
    detection_summary,
    dice,
    iou,
    match_detections,
    roc_auc,
    wilson_ci,
)


def square(h, w, y0, x0, s):
    m = np.zeros((h, w), np.uint8)
    m[y0 : y0 + s, x0 : x0 + s] = 1
    return m


def test_dice_iou_identical_and_disjoint():
    a = square(64, 64, 10, 10, 20)
    b = square(64, 64, 40, 40, 20)
    assert dice(a, a) == pytest.approx(1.0)
    assert iou(a, a) == pytest.approx(1.0)
    assert dice(a, b) == 0.0
    assert iou(a, b) == 0.0


def test_dice_iou_known_overlap():
    a = square(64, 64, 0, 0, 20)  # 400 px
    b = square(64, 64, 0, 10, 20)  # 400 px, ortak 200 px
    assert dice(a, b) == pytest.approx(2 * 200 / 800)
    assert iou(a, b) == pytest.approx(200 / 600)


def test_empty_masks_count_as_perfect():
    z = np.zeros((8, 8), np.uint8)
    assert dice(z, z) == 1.0
    assert iou(z, z) == 1.0


def test_components_drop_tiny_blobs_and_score_by_mean_prob():
    m = square(100, 100, 10, 10, 20)
    m[80, 80] = 1  # tek piksel gurultu
    prob = np.where(m > 0, 0.9, 0.0)
    comps = components(m, prob)
    assert len(comps) == 1
    assert comps[0].box == (10, 10, 30, 30)
    assert comps[0].score == pytest.approx(0.9)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_matching_one_gt_two_preds_gives_one_tp_one_fp():
    gt = [Component((0, 0, 10, 10), 100)]
    preds = [Component((0, 0, 10, 10), 100, 0.9), Component((1, 0, 11, 10), 100, 0.8)]
    r = match_detections(preds, gt)
    assert r.n_gt_hit == 1
    assert r.scored == [(0.9, True), (0.8, False)]


def test_ap_perfect_and_worst():
    perfect = [DetectionResult([(0.9, True)], 1, 1), DetectionResult([(0.8, True)], 1, 1)]
    assert average_precision(perfect) == pytest.approx(1.0)
    missed = [DetectionResult([], 1, 0)]
    assert average_precision(missed) == 0.0


def test_ap_ranking_matters():
    # yuksek skorlu yanlis tespit once gelirse AP 1'in altina duser
    good_first = [DetectionResult([(0.9, True), (0.1, False)], 1, 1)]
    bad_first = [DetectionResult([(0.9, False), (0.1, True)], 1, 1)]
    assert average_precision(good_first) == pytest.approx(1.0)
    assert average_precision(bad_first) == pytest.approx(0.5)


def test_detection_summary_counts():
    rs = [DetectionResult([(0.9, True), (0.5, False)], 1, 1), DetectionResult([], 2, 0)]
    s = detection_summary(rs)
    assert s["lesion_sensitivity"] == pytest.approx(1 / 3)
    assert s["precision"] == pytest.approx(0.5)
    assert s["fp_per_image"] == pytest.approx(0.5)
    assert s["n_gt_polyps"] == 3


def test_wilson_ci_contains_point_and_is_bounded():
    lo, hi = wilson_ci(95, 100)
    assert 0.0 <= lo < 0.95 < hi <= 1.0
    lo0, hi0 = wilson_ci(0, 20)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0 < hi0 < 0.2


def test_roc_auc_extremes_and_ties():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)
    assert roc_auc([0.1], [0.9]) == pytest.approx(0.0)
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)
