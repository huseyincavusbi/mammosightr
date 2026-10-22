#!/usr/bin/env python3
"""YARISMA DICOM -> 8-bit breast-crop PNG (768x1024) matching the Gulluk/RSNA parquet style.

Pipeline (reverse-engineered from code/rsna-crop + parquet audit):
  DICOM decode -> VOI windowing (WW/WC, fallback min-max) -> MONOCHROME1 invert
  -> 8-bit -> tight ROI crop (threshold>10, largest connected component, 5px margin)
  -> aspect-preserving resize to fit 768x1024 -> centered on black canvas -> PNG

Reads directly from the raw split zips (no extraction), resumable, writes a manifest.

Examples:
  tools/preprocess_yarisma.py --limit 8          # smoke test (2 cases)
  tools/preprocess_yarisma.py                    # full 8,000 images
"""
import argparse
import csv
import glob
import io
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_set import MultiPart

ROOT = Path(__file__).resolve().parents[1]
PARTS = sorted(glob.glob(str(ROOT / "data/YARISMA/MG_YARISMA.zip.*")))
OUT = ROOT / "yarisma"
TARGET_W, TARGET_H = 768, 1024
MIN_AREA_FRAC = 0.04


def window_to_uint8(ds):
    arr = ds.pixel_array.astype(np.float32)
    ww, wc = getattr(ds, "WindowWidth", None), getattr(ds, "WindowCenter", None)
    if ww is not None and wc is not None:
        ww = float(ww[0]) if hasattr(ww, "__getitem__") else float(ww)
        wc = float(wc[0]) if hasattr(wc, "__getitem__") else float(wc)
        func = str(getattr(ds, "VOILUTFunction", "LINEAR")).upper()
        if func == "SIGMOID":
            out = 255.0 / (1.0 + np.exp((arr - wc) * (-4.0 / ww)))
        else:
            if func == "LINEAR":
                wc, ww = wc - 0.5, ww - 1.0
            out = np.clip(arr * (255.0 / ww) + (-wc / ww + 0.5) * 255.0, 0, 255)
    else:
        mn, mx = float(arr.min()), float(arr.max())
        out = (arr - mn) / (mx - mn + 1e-6) * 255.0
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        out = 255.0 - out
    return out.astype(np.uint8)


def crop_roi(img):
    h, w = img.shape
    if h <= 10 or w <= 10:
        return img, None
    m = img[5:-5, 5:-5]
    n, _, stats, _ = cv2.connectedComponentsWithStats((m > 10).astype(np.uint8), 8, cv2.CV_32S)
    if n <= 1:
        return img, None
    idx = 1 + int(stats[1:, 4].argmax())
    x, y, bw, bh, area = stats[idx]
    if area < MIN_AREA_FRAC * m.size:
        return img, None
    x0, y0 = max(0, x - 5), max(0, y - 5)
    x1, y1 = min(w, x + bw + 5), min(h, y + bh + 5)
    return img[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def fit_pad(img):
    h, w = img.shape
    s = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    x0, y0 = (TARGET_W - nw) // 2, (TARGET_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, (nw, nh)


def parse_name(name):
    parts = name.split("/")
    patient = parts[-2]
    fn = parts[-1]
    return patient, fn.replace(".dcm", ".png"), fn[1:-4], fn[0]


def process_file(data, out_path):
    ds = pydicom.dcmread(io.BytesIO(data))
    img = window_to_uint8(ds)
    crop, box = crop_roi(img)
    canvas, size = fit_pad(crop)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    return ok, box, size, canvas.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--manifest", default=str(ROOT / "labels/yarisma_manifest.csv"))
    args = ap.parse_args()
    out_root = Path(args.out)

    mp = MultiPart(PARTS)
    zf = zipfile.ZipFile(mp)
    names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
    names.sort()
    if args.limit:
        names = names[: args.limit]
    print(f"processing {len(names)} files from {len(PARTS)} parts")

    rows = []
    fails = 0
    start_t = time.time()
    done = 0
    total = len(names)
    for i, name in enumerate(names, 1):
        patient, fn, view, lat = parse_name(name)
        dst = out_root / patient / fn
        if dst.exists():
            done += 1
            continue
        try:
            data = zf.read(name)
            ok, box, size, shape = process_file(data, dst)
            rows.append([patient, fn, view, lat, *box] if box else [patient, fn, view, lat, "", "", "", ""])
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
        done += 1
        if i % 100 == 0 or i == total:
            el = time.time() - start_t
            rate = done / el if el > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"[{i}/{total}] {rate:.1f} img/s | elapsed {el/60:.1f} min | ETA {eta/60:.1f} min | fails {fails}", flush=True)
    el = time.time() - start_t
    print(f"total time {el/60:.1f} min ({done/el:.1f} img/s)", flush=True)

    if rows:
        mpath = Path(args.manifest)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        write_header = not mpath.exists()
        with open(mpath, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["patient_id", "image_filename", "view", "laterality", "x0", "y0", "x1", "y1"])
            w.writerows(rows)
    print(f"done: {len(rows)} written, {fails} failed")


if __name__ == "__main__":
    main()
