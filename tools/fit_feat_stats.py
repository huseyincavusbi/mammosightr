"""Recover the exact feature standardization stats from cached z-scored features.

For paired raw features r and cached z = (r - mu)/sd, fit per-dim r = mu + sd*z.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import timm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_CASES = 128


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    z = np.load(ROOT / "notebooks/out/features_full_gulluk_convnext_small.npz")
    keys = [k for k in z.keys() if k.startswith("tr_")]
    rng = np.random.default_rng(0)
    rng.shuffle(keys)
    pids = [k.split("_", 1)[1] for k in keys[:N_CASES]]

    img = pd.read_csv(ROOT / "labels/images.csv")
    img["patient_id"] = img.patient_id.astype(str)
    img = img[img.patient_id.isin(pids)].sort_values(["shard", "row_in_shard"])

    m = timm.create_model("convnext_small", pretrained=False)
    m.head.fc = nn.Linear(768, 768)
    m.birads_layer = nn.Sequential(nn.Linear(768, 5))
    m.density_layer = nn.Sequential(nn.Linear(768, 4))
    m.load_state_dict(torch.load(ROOT / "code/MAM-CLIP/weights/convnext_small_ftune.pth",
                                 map_location="cpu", weights_only=False))
    m = m.eval().to(device)

    raw_all, z_all = [], []
    for shard, g in img.groupby("shard"):
        p = ROOT / "preprocessed/data" / shard
        t = pq.read_table(p, columns=["image"])
        col = t.column("image")
        for pid, gg in g.groupby("patient_id"):
            rows = [int(r) for r in gg.row_in_shard]
            xs = []
            for r in rows:
                im = Image.open(io.BytesIO(col[r].as_py()["bytes"])).convert("L")
                a = np.array(im).astype(np.float32)
                a = a - a.min()
                a = a / max(float(a.max()), 1e-6)
                xs.append(np.stack([a, a, a]))
            with torch.no_grad():
                f = m.forward_head(m.forward_features(torch.from_numpy(np.stack(xs)).to(device)),
                                   pre_logits=True).float().cpu().numpy()
            zz = z[f"tr_{pid}"]
            if f.shape == zz.shape:
                raw_all.append(f)
                z_all.append(zz)
        print(f"  shard {shard}: done", flush=True)

    raw = np.concatenate(raw_all, 0)
    zz = np.concatenate(z_all, 0)
    zc = zz - zz.mean(0)
    sd = (zc * (raw - raw.mean(0))).sum(0) / (zc * zc).sum(0)
    mu = raw.mean(0) - sd * zz.mean(0)
    pred = mu + sd * zz
    ss_res = ((raw - pred) ** 2).sum()
    ss_tot = ((raw - raw.mean(0)) ** 2).sum()
    print(f"n={len(raw)} | R2={1 - ss_res / ss_tot:.6f}")
    np.savez(ROOT / "results/feat_stats.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32), n=len(raw))
    print("saved results/feat_stats.npz")


if __name__ == "__main__":
    main()
