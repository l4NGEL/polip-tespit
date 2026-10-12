"""Veri setlerini data/ altina indirir.

Kvasir-SEG   : egitim + ic test (Simula, resmi sunucu)
CVC-ClinicDB : dis test, egitimde hic kullanilmaz (HuggingFace dosya aynasi,
               orijinal kaynak: https://polyp.grand-challenge.org/CVCClinicDB/)
normal-cecum : polipsiz kolon goruntuleri, yalnizca ozgulluk testi (Kvasir v2, HF aynasi)

Kullanim:  python scripts/download_data.py
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
CVC_REPO = "berkaytrhn/cvc-clinicdb"
NORMAL_REPO = "San-D/Kvasir_V2"


def download_kvasir() -> None:
    target = DATA / "kvasir-seg"
    if (target / "images").is_dir() and len(list((target / "images").glob("*.jpg"))) == 1000:
        print("Kvasir-SEG zaten var")
        return
    zip_path = DATA / "kvasir-seg.zip"
    if not zip_path.exists():
        print(f"indiriliyor: {KVASIR_URL}")
        urllib.request.urlretrieve(KVASIR_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA / "_kvasir_tmp")
    # zip icinde "Kvasir-SEG/images", "Kvasir-SEG/masks", "Kvasir-SEG/kavsir_bboxes.json"
    src = next((DATA / "_kvasir_tmp").glob("*/images")).parent
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    shutil.rmtree(DATA / "_kvasir_tmp")
    zip_path.unlink()
    n = len(list((target / "images").glob("*.jpg")))
    print(f"Kvasir-SEG: {n} goruntu")


def download_cvc() -> None:
    target = DATA / "cvc-clinicdb"
    if (target / "Original").is_dir():
        print("CVC-ClinicDB zaten var")
        return
    from huggingface_hub import snapshot_download

    tmp = Path(snapshot_download(repo_id=CVC_REPO, repo_type="dataset", local_dir=DATA / "_cvc_tmp"))
    src = tmp / "CVC-ClinicDB"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    shutil.rmtree(tmp)
    n = len(list((target / "Original").iterdir()))
    print(f"CVC-ClinicDB: {n} goruntu")


def download_normal_cecum() -> None:
    """Polipsiz kolon goruntuleri (Kvasir v2 'normal-cecum'; aynada 100 adet): yalnizca goruntu duzeyi
    ozgulluk icin test negatifi. Resmi kaynak 2.5 GB'lik tek zip oldugu icin sinif klasorunu
    ayri tutan HF aynasindan yalnizca bu sinif indirilir. Ayna sinif basina 100
    goruntuluk bir alt kume; ozgulluk bu yuzden 100 negatif uzerinden olculur."""
    target = DATA / "normal-cecum"
    if target.is_dir() and len(list(target.glob("*.jpg"))) >= 100:
        print("normal-cecum zaten var")
        return
    from huggingface_hub import snapshot_download

    tmp = Path(snapshot_download(repo_id=NORMAL_REPO, repo_type="dataset", local_dir=DATA / "_normal_tmp",
                                 allow_patterns=["kvasir-dataset-v2/normal-cecum/*"]))
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(tmp / "kvasir-dataset-v2" / "normal-cecum"), str(target))
    shutil.rmtree(tmp)
    print(f"normal-cecum: {len(list(target.glob('*.jpg')))} goruntu")


def main() -> int:
    DATA.mkdir(exist_ok=True)
    download_kvasir()
    download_cvc()
    download_normal_cecum()
    return 0


if __name__ == "__main__":
    sys.exit(main())
