"""Estimate per-dim feature mean/std for inference-time standardization."""
import glob
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import timm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N = 1024
BATCH = 16


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    m = timm.create_model("convnext_small", pretrained=False)
    m.head.fc = nn.Linear(768, 768)
    m.birads_layer = nn.Sequential(nn.Linear(768, 5))
    m.density_layer = nn.Sequential(nn.Linear(768, 4))
    m.load_state_dict(torch.load(ROOT / "code/MAM-CLIP/weights/convnext_small_ftune.pth",
                                 map_location="cpu", weights_only=False))
    m = m.eval().to(device)

    shard = sorted(glob.glob(str(ROOT / "preprocessed/data/*.parquet")))[0]
    t = pq.read_table(shard, columns=["image"]).slice(0, N)
    col = t.column("image")
    feats = []
    with torch.no_grad():
        for start in range(0, len(col), BATCH):
            xs = []
            for i in range(start, min(start + BATCH, len(col))):
                im = Image.open(io.BytesIO(col[i].as_py()["bytes"])).convert("L")
                a = np.array(im).astype(np.float32)
                a = a - a.min()
                a = a / max(float(a.max()), 1e-6)
                xs.append(np.stack([a, a, a]))
            x = torch.from_numpy(np.stack(xs)).to(device)
            f = m.forward_head(m.forward_features(x), pre_logits=True)
            feats.append(f.float().cpu().numpy())
            print(f"  {min(start + BATCH, len(col))}/{len(col)}", flush=True)
    arr = np.concatenate(feats, 0)
    mu = arr.mean(0).astype(np.float32)
    sd = (arr.std(0) + 1e-6).astype(np.float32)
    out = ROOT / "results/feat_stats.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, mu=mu, sd=sd, n=len(arr))
    print(f"saved {out} | n={len(arr)} | mu[0..3]={mu[:4]} | sd[0..3]={sd[:4]}")


if __name__ == "__main__":
    main()
