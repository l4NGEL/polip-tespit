"""ONNX'e aktarim, PyTorch ile cikti uyumu ve gecikme olcumu.

Olculen yollar (batch 1, 352x352): PyTorch FP32 / FP16 (CUDA), ONNX Runtime CPU, CUDA ve
(kuruluysa) TensorRT FP16. Her yol icin medyan ve p95 gecikme; ONNX yollari icin PyTorch FP32'ye
gore en buyuk mutlak olasilik farki ve maske uyumu (Dice) raporlanir.

Kullanim:  python scripts/export_onnx.py --ckpt checkpoints/unet_r34/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polip.data import cvc_samples, read_pair  # noqa: E402
from polip.metrics import dice  # noqa: E402
from polip.model import INPUT_SIZE, load_checkpoint  # noqa: E402
from polip.data import normalize  # noqa: E402

import cv2  # noqa: E402


def timed(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "fps": float(1000 / np.median(ts))}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, default=Path("checkpoints/unet_r34/model.onnx"))
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--parity-images", type=int, default=50)
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt, "cuda")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    torch.onnx.export(model, dummy, str(args.onnx), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    print(f"ONNX yazildi: {args.onnx} ({args.onnx.stat().st_size / 1e6:.1f} MB)")

    # gercek goruntulerle (CVC'den ilk N) girdi hazirla
    xs = []
    for s in cvc_samples()[: args.parity_images]:
        img, _ = read_pair(s)
        xs.append(normalize(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)))[None])
    with torch.no_grad():
        ref = [torch.sigmoid(model(torch.from_numpy(x).cuda())).cpu().numpy() for x in xs]

    report = {"input": [1, 3, INPUT_SIZE, INPUT_SIZE], "gpu": torch.cuda.get_device_name(0), "paths": {}}
    x0 = torch.from_numpy(xs[0]).cuda()

    def torch_fp32():
        with torch.no_grad():
            model(x0)
        torch.cuda.synchronize()

    def torch_fp16():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(x0)
        torch.cuda.synchronize()

    report["paths"]["pytorch_fp32_cuda"] = timed(torch_fp32, 20, args.runs)
    report["paths"]["pytorch_fp16_cuda"] = timed(torch_fp16, 20, args.runs)

    providers = {
        "onnxruntime_cpu": ["CPUExecutionProvider"],
        "onnxruntime_cuda": ["CUDAExecutionProvider"],
        "onnxruntime_tensorrt_fp16": [("TensorrtExecutionProvider", {
            "trt_fp16_enable": True, "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(args.onnx.parent / "trt_cache")}), "CUDAExecutionProvider"],
    }
    for name, prov in providers.items():
        try:
            sess = ort.InferenceSession(str(args.onnx), providers=prov)
        except Exception as e:  # TensorRT kutuphaneleri yoksa burada duser
            report["paths"][name] = {"error": str(e).splitlines()[0][:200]}
            print(f"{name}: kullanilamadi ({report['paths'][name]['error']})")
            continue
        active = sess.get_providers()[0]
        wanted = prov[0] if isinstance(prov[0], str) else prov[0][0]
        if active != wanted:
            report["paths"][name] = {"error": f"istenen {wanted}, etkin {active}"}
            print(f"{name}: {wanted} yuklenemedi, {active} devrede -- olculmedi")
            continue
        outs = [sigmoid(sess.run(None, {"image": x})[0]) for x in xs]
        max_diff = max(float(np.abs(o - r).max()) for o, r in zip(outs, ref))
        mask_dice = float(np.mean([dice(o > 0.5, r > 0.5) for o, r in zip(outs, ref)]))
        runs = args.runs if name != "onnxruntime_cpu" else max(20, args.runs // 10)
        t = timed(lambda: sess.run(None, {"image": xs[0]}), 10, runs)
        report["paths"][name] = {**t, "max_abs_prob_diff_vs_torch": max_diff, "mask_dice_vs_torch": mask_dice}
        print(f"{name}: medyan {t['median_ms']:.2f} ms ({t['fps']:.0f} FPS)  max|dp| {max_diff:.2e}  maske Dice {mask_dice:.4f}")

    for k in ("pytorch_fp32_cuda", "pytorch_fp16_cuda"):
        t = report["paths"][k]
        print(f"{k}: medyan {t['median_ms']:.2f} ms ({t['fps']:.0f} FPS)")
    out = Path("results") / "onnx_benchmark.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
