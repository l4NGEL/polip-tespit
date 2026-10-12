"""Model tanimi ve tahmin yardimcilari."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from .data import normalize

INPUT_SIZE = 352


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def load_checkpoint(path: Path, device: str = "cuda") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class UnetPredictor:
    """predict(img) -> (olasilik haritasi, ikili maske); esik 0.5."""

    threshold = 0.5

    def __init__(self, ckpt: Path, device: str = "cuda"):
        self.model, self.device = load_checkpoint(ckpt, device), device

    def predict(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob = predict_prob(self.model, img_rgb, self.device)
        return prob, (prob > self.threshold).astype(np.uint8)


class YoloPredictor:
    """YOLOv8-seg ornek maskelerini tek haritaya indirger.

    Olasilik haritasi = pikseli kapsayan orneklerin en yuksek guveni; ikili maske = guveni
    esigi gecen orneklerin birlesimi. Esik 0.25 (Ultralytics varsayilani), testte ayarlanmaz.
    """

    threshold = 0.25

    def __init__(self, weights: Path, imgsz: int = INPUT_SIZE):
        from ultralytics import YOLO

        self.model, self.imgsz = YOLO(str(weights)), imgsz

    def predict(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_rgb.shape[:2]
        r = self.model.predict(img_rgb[..., ::-1], imgsz=self.imgsz, conf=0.01, retina_masks=True, verbose=False)[0]
        prob = np.zeros((h, w), np.float32)
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for m, c in zip(masks, confs):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) if m.shape != (h, w) else m
                prob = np.where(m > 0.5, np.maximum(prob, c), prob)
        return prob, (prob >= self.threshold).astype(np.uint8)


@torch.no_grad()
def predict_prob(model: torch.nn.Module, img_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Tek goruntu icin polip olasilik haritasi, ozgun cozunurlukte (HxW float32)."""
    h, w = img_rgb.shape[:2]
    x = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(normalize(x))[None].to(device)
    prob = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
