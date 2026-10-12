"""CVC-ClinicDB kare dizisini (29 kolonoskopi videosundan sirali kareler) ONNX Runtime ile isleyip
FFmpeg'e boru hattiyla verir: H.264 MP4 + README icin kisa GIF.

Yesil kontur: uzman maskesi, kirmizi dolgu: model tahmini. Ust bilgi: kare no, model gecikmesi.
CVC egitimde hic kullanilmadigi icin video dis test verisidir.

Kullanim:  python scripts/make_demo_video.py --onnx checkpoints/unet_r34/model.onnx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polip.data import cvc_samples, normalize, read_pair  # noqa: E402
from polip.model import INPUT_SIZE  # noqa: E402


def render(img_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, text: str) -> np.ndarray:
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    red = np.zeros_like(out)
    red[..., 2] = 255
    m = pred.astype(bool)
    out[m] = (0.6 * out[m] + 0.4 * red[m]).astype(np.uint8)
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (0, 220, 0), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/demo_cvc.mp4"))
    ap.add_argument("--gif", type=Path, default=Path("docs/demo_cvc.gif"))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif-frames", default="1:60", help="GIF icin kare araligi (1 tabanli, dahil)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(args.onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    samples = cvc_samples()
    h, w = read_pair(samples[0])[0].shape[:2]
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )
    lat = []
    for i, s in enumerate(samples, 1):
        img, gt = read_pair(s)
        x = normalize(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)))[None]
        t0 = time.perf_counter()
        logits = sess.run(None, {"image": x})[0][0, 0]
        lat.append((time.perf_counter() - t0) * 1000)
        prob = cv2.resize(1 / (1 + np.exp(-logits)), (w, h))
        frame = render(img, gt, prob > 0.5, f"CVC-ClinicDB kare {i}/{len(samples)}  |  model {lat[-1]:.1f} ms  |  yesil: uzman, kirmizi: model")
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB), medyan gecikme {np.median(lat):.1f} ms ({sess.get_providers()[0]})")

    a, b = (int(v) for v in args.gif_frames.split(":"))
    start, dur = (a - 1) / args.fps, (b - a + 1) / args.fps
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", str(args.out),
         "-vf", "fps=8,scale=384:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer",
         str(args.gif)],
        check=True,
    )
    print(f"{args.gif}  ({args.gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
