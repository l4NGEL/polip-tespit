"""Veri okuma, bolme ve artirma."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SPLIT_FILE = ROOT / "splits" / "kvasir_split.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    name: str
    image: Path
    mask: Path


def kvasir_samples() -> list[Sample]:
    d = DATA / "kvasir-seg"
    return [Sample(p.stem, p, d / "masks" / p.name) for p in sorted((d / "images").glob("*.jpg"))]


def cvc_samples() -> list[Sample]:
    d = DATA / "cvc-clinicdb"
    imgs = sorted((d / "Original").glob("*.png"), key=lambda p: int(p.stem))
    return [Sample(p.stem, p, d / "Ground Truth" / p.name) for p in imgs]


def make_kvasir_split(seed: int = 42, n_val: int = 100, n_test: int = 100) -> dict[str, list[str]]:
    """Sabit tohumla 800/100/100 bolme; bir kez uretilip splits/ altina yazilir ve sonra hep o okunur."""
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    names = [s.name for s in kvasir_samples()]
    rng = random.Random(seed)
    rng.shuffle(names)
    split = {
        "test": sorted(names[:n_test]),
        "val": sorted(names[n_test : n_test + n_val]),
        "train": sorted(names[n_test + n_val :]),
    }
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1))
    return split


def kvasir_subset(part: str) -> list[Sample]:
    keep = set(make_kvasir_split()[part])
    return [s for s in kvasir_samples() if s.name in keep]


def read_pair(s: Sample) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(cv2.imread(str(s.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(s.mask), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        raise FileNotFoundError(s)
    # Ultralytics import edilince cv2.imread'i kendi surumuyle degistirir ve gri goruntuyu
    # (H, W, 1) dondurur; ayni surecte YOLO yuklu olsa da maske hep 2B kalsin.
    if m.ndim == 3:
        m = m[..., 0]
    return img, (m > 127).astype(np.uint8)


def normalize(img_rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> 3xHxW float32."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def augment(img: np.ndarray, m: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5:
        img, m = img[::-1], m[::-1]
    k = int(rng.integers(4))
    img, m = np.rot90(img, k), np.rot90(m, k)
    h, w = m.shape
    # olcek + kaydirma + kucuk aci
    ang = float(rng.uniform(-20, 20))
    sc = float(rng.uniform(0.8, 1.25))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-0.08, 0.08, size=2) * (w, h)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(np.ascontiguousarray(m), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    # parlaklik / kontrast / renk tonu
    x = img.astype(np.float32)
    x = x * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    x = x * rng.uniform(0.92, 1.08, size=3)
    img = np.clip(x, 0, 255).astype(np.uint8)
    return img, m


class PolypDataset(Dataset):
    def __init__(self, samples: list[Sample], size: int = 352, train: bool = False):
        self.samples, self.size, self.train = samples, size, train
        self.rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img, m = read_pair(self.samples[i])
        if self.train:
            # DataLoader her isciye ve her epoka farkli torch tohumu verir; numpy uretecini ondan turet,
            # yoksa Windows'ta kopyalanan isciler ayni artirma dizisini uretir
            if self.rng is None:
                self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
            img, m = augment(img, m, self.rng)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(normalize(img)), torch.from_numpy(m[None].astype(np.float32))
