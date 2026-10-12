import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from polip.data import SPLIT_FILE, augment, normalize

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SPLIT_FILE.exists(), reason="bolme dosyasi henuz uretilmedi")
def test_split_is_disjoint_and_complete():
    import json

    s = json.loads(SPLIT_FILE.read_text())
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert (len(tr), len(va), len(te)) == (800, 100, 100)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_augment_keeps_mask_aligned_with_image():
    # goruntu ve maske ayni karede: artirmadan sonra maskeli bolge hala "beyaz" pikseller olmali
    rng = np.random.default_rng(0)
    img = np.zeros((120, 160, 3), np.uint8)
    m = np.zeros((120, 160), np.uint8)
    img[30:70, 50:100] = 255
    m[30:70, 50:100] = 1
    for _ in range(20):
        a_img, a_m = augment(img, m, rng)
        assert a_img.shape[:2] == a_m.shape
        inside = a_img[a_m > 0].mean()
        outside = a_img[a_m == 0].mean() if (a_m == 0).any() else 0
        assert inside > outside + 100


def test_augment_mask_stays_binary():
    rng = np.random.default_rng(1)
    m = np.zeros((64, 64), np.uint8)
    m[10:40, 10:40] = 1
    _, a_m = augment(np.zeros((64, 64, 3), np.uint8), m, rng)
    assert set(np.unique(a_m)) <= {0, 1}


def test_read_pair_mask_is_2d_even_if_imread_returns_channel_axis(tmp_path, monkeypatch):
    # Ultralytics cv2.imread'i (H, W, 1) donduren bir surumle degistirir; read_pair bunu tolere etmeli
    from polip import data as d

    img = np.full((20, 30, 3), 100, np.uint8)
    m = np.zeros((20, 30), np.uint8)
    m[5:10, 5:15] = 255
    cv2.imwrite(str(tmp_path / "i.png"), img)
    cv2.imwrite(str(tmp_path / "m.png"), m)
    real = cv2.imread

    def patched(path, flags=cv2.IMREAD_COLOR):
        out = real(path, flags)
        return out[..., None] if out is not None and out.ndim == 2 else out

    monkeypatch.setattr(d.cv2, "imread", patched)
    _, mask = d.read_pair(d.Sample("x", tmp_path / "i.png", tmp_path / "m.png"))
    assert mask.shape == (20, 30)
    assert mask.sum() == 50


def test_normalize_shape_and_dtype():
    x = normalize(np.full((10, 12, 3), 128, np.uint8))
    assert x.shape == (3, 10, 12) and x.dtype == np.float32


def test_yolo_polygon_roundtrip_recovers_mask():
    mod = _load_script("yolo_prepare")
    m = np.zeros((200, 300), np.uint8)
    cv2.circle(m, (100, 80), 40, 1, -1)
    cv2.rectangle(m, (200, 120), (260, 180), 1, -1)
    m[5, 5] = 1  # tek piksel: atilmali
    polys = mod.mask_to_polygons(m)
    assert len(polys) == 2
    assert all(((p >= 0) & (p <= 1)).all() for p in polys)
    back = np.zeros_like(m)
    for p in polys:
        cv2.fillPoly(back, [np.round(p * (300, 200)).astype(np.int32)], 1)
    m[5, 5] = 0
    inter = np.logical_and(back, m).sum()
    assert inter / np.logical_or(back, m).sum() > 0.95
