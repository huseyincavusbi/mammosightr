"""Extract frozen-backbone features for train/val/YARISMA and save standardized npz.

Reproduces the cached features used by the head-training scripts:
  train = folds != 1, val = fold 1, test = YARISMA.
Per-image preprocessing: grayscale, min-max to [0,1], three channels, 768x1024.
Standardization uses the mean/std of the training split only.

Example:
  python tools/extract_features.py \
    --parquet-dir data/parquet --yarisma-dir yarisma \
    --labels-dir assets/labels --weights weights/convnext_small_ftune.pth \
    --out features/features_full_gulluk_convnext_small.npz
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import timm
from PIL import Image


def preprocess(img):
    a = np.array(img.convert("L"))
    if a.shape != (1024, 768):
        import cv2
        a = cv2.resize(a, (768, 1024))
    x = a.astype(np.float32)
    x = x - x.min()
    x = x / max(float(x.max()), 1e-6)
    return np.stack([x, x, x])


def load_backbone(weights, device):
    m = timm.create_model("convnext_small", pretrained=False)
    m.head.fc = torch.nn.Linear(768, 768)
    m.birads_layer = torch.nn.Sequential(torch.nn.Linear(768, 5))
    m.density_layer = torch.nn.Sequential(torch.nn.Linear(768, 4))
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
    return m.eval().to(device)


@torch.no_grad()
def forward(model, imgs, device):
    x = torch.from_numpy(np.stack([preprocess(im) for im in imgs])).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        f = model.forward_head(model.forward_features(x), pre_logits=True)
    return f.float().cpu().numpy()


def extract_parquet(model, rows_by_pid, parquet_dir, device, batch_cases, tag):
    """rows_by_pid: pid -> list of (shard, row). Reads shard image columns lazily."""
    features = {}
    by_shard = {}
    for pid, rows in rows_by_pid.items():
        for shard, row in rows:
            by_shard.setdefault(shard, []).append((pid, row))
    done = 0
    total = len(rows_by_pid)
    for shard, pairs in by_shard.items():
        col = pq.read_table(Path(parquet_dir) / shard, columns=["image"]).column("image")
        by_pid = {}
        for pid, row in pairs:
            by_pid.setdefault(pid, []).append(int(row))
        pids = list(by_pid)
        for start in range(0, len(pids), batch_cases):
            chunk = pids[start:start + batch_cases]
            imgs, owners = [], []
            for pid in chunk:
                for row in by_pid[pid]:
                    imgs.append(Image.open(io.BytesIO(col[row].as_py()["bytes"])).convert("L"))
                    owners.append(pid)
            feats = forward(model, imgs, device)
            offset = 0
            for pid in chunk:
                n = len(by_pid[pid])
                features[pid] = feats[offset:offset + n]
                offset += n
            done += len(chunk)
            if done % 25 == 0 or done == total:
                print(f"  {tag}: {done}/{total} cases", flush=True)
        del col
    return features


def extract_yarisma(model, pids, manifest, yarisma_dir, device, batch_cases):
    features = {}
    done = 0
    for start in range(0, len(pids), batch_cases):
        chunk = pids[start:start + batch_cases]
        imgs = []
        for pid in chunk:
            g = manifest[manifest.patient_id == pid]
            for fn in g.image_filename:
                imgs.append(Image.open(Path(yarisma_dir) / pid / fn).convert("L"))
        feats = forward(model, imgs, device)
        offset = 0
        for pid in chunk:
            n = len(manifest[manifest.patient_id == pid])
            features[pid] = feats[offset:offset + n]
            offset += n
        done += len(chunk)
        if done % 25 == 0 or done == len(pids):
            print(f"  yarisma: {done}/{len(pids)} cases", flush=True)
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, help="directory with train-*.parquet shards")
    ap.add_argument("--yarisma-dir", required=True, help="directory with <patient>/<VIEW>.png")
    ap.add_argument("--labels-dir", required=True, help="directory with images.csv, folds.csv, yarisma_manifest.csv")
    ap.add_argument("--weights", required=True, help="ConvNeXt-small fine-tuned checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-cases", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    labels = Path(args.labels_dir)
    print(f"device: {device}")

    images = pd.read_csv(labels / "images.csv")
    images["patient_id"] = images.patient_id.astype(str)
    images = images.sort_values(["shard", "row_in_shard"])
    folds = pd.read_csv(labels / "folds.csv")
    folds["patient_id"] = folds.patient_id.astype(str)
    fold_of = dict(zip(folds.patient_id, folds.fold))
    manifest = pd.read_csv(labels / "yarisma_manifest.csv")
    manifest["patient_id"] = manifest.patient_id.astype(str)

    rows_by_pid = {}
    for row in images.itertuples():
        rows_by_pid.setdefault(row.patient_id, []).append((row.shard, int(row.row_in_shard)))

    train_pids = [p for p in rows_by_pid if fold_of.get(p) != 1]
    val_pids = [p for p in rows_by_pid if fold_of.get(p) == 1]
    test_pids = list(dict.fromkeys(manifest.patient_id))
    print(f"train {len(train_pids)} | val {len(val_pids)} | test {len(test_pids)}")

    model = load_backbone(args.weights, device)
    tr = extract_parquet(model, {p: rows_by_pid[p] for p in train_pids}, args.parquet_dir,
                         device, args.batch_cases, "train")
    va = extract_parquet(model, {p: rows_by_pid[p] for p in val_pids}, args.parquet_dir,
                         device, args.batch_cases, "val")
    te = extract_yarisma(model, test_pids, manifest, args.yarisma_dir, device, args.batch_cases)

    all_train = np.concatenate(list(tr.values()), 0)
    mu = all_train.mean(0).astype(np.float32)
    sd = (all_train.std(0) + 1e-6).astype(np.float32)

    def std(d):
        return {pid: ((v - mu) / sd).astype(np.float32) for pid, v in d.items()}

    tr, va, te = std(tr), std(va), std(te)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{f"tr_{p}": v for p, v in tr.items()},
                        **{f"va_{p}": v for p, v in va.items()},
                        **{f"te_{p}": v for p, v in te.items()})
    np.savez(out.parent / "feat_stats.npz", mu=mu, sd=sd, n=len(all_train))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) + feat_stats.npz")


if __name__ == "__main__":
    main()
