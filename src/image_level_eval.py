"""Image-level BI-RADS evaluation on YARISMA (Gulluk-style protocol).

Train a small MLP on cached per-view features with image-level labels,
evaluate on held-out fold images (in-domain) and YARISMA images (external).
CPU-only; no GPU contention.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("MAMMO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from train_heads import load_view_meta  # noqa: E402

B3 = {"0": 0, "1-2": 1, "4-5": 2}
B3_IMG = {0: 0, 1: 1, 2: 1, 4: 2, 5: 2}
FEAT = ROOT / "features/features_full_gulluk_convnext_small.npz"


def log(*a):
    print(*a, flush=True)


class MLP(nn.Module):
    def __init__(self, dim=768, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 3))

    def forward(self, x):
        return self.net(x)


def macro_f1(y, p):
    f1 = []
    for c in (0, 1, 2):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1)), [round(x, 4) for x in f1]


def main():
    torch.set_num_threads(8)
    device = torch.device("cpu")
    z = np.load(FEAT)
    img = pd.read_csv(ROOT / "labels/images.csv")
    img["patient_id"] = img.patient_id.astype(str)
    img = img.sort_values(["shard", "row_in_shard"])
    by_pid = {pid: g for pid, g in img.groupby("patient_id")}

    man = pd.read_csv(ROOT / "labels/yarisma_manifest.csv")
    man["patient_id"] = man.patient_id.astype(str)
    by_test = {pid: g for pid, g in man.groupby("patient_id", sort=False)}
    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str); cases = cases.set_index("patient_id")

    def split_arrays(keys):
        X, y = [], []
        for k in keys:
            pid = k.split("_", 1)[1]
            g = by_pid[pid]
            X.append(z[k])
            y += [B3_IMG[int(v)] for v in g.birads5_img]
        return np.concatenate(X, 0).astype(np.float32), np.array(y)

    tr_keys = [k for k in z.keys() if k.startswith("tr_")]
    va_keys = [k for k in z.keys() if k.startswith("va_")]
    te_keys = [k for k in z.keys() if k.startswith("te_")]
    Xtr, ytr = split_arrays(tr_keys)
    Xva, yva = split_arrays(va_keys)
    Xte, yte = [], []
    for k in te_keys:
        pid = k.split("_", 1)[1]
        Xte.append(z[k])
        yte += [B3[str(cases.loc[pid, "birads3"])]] * len(by_test[pid])
    Xte = np.concatenate(Xte, 0).astype(np.float32); yte = np.array(yte)
    log(f"images: train {len(Xtr)} val {len(Xva)} test {len(Xte)}")

    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=3).clip(1)
    w = torch.tensor(len(ytr) / (3 * cnt), dtype=torch.float32)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    for ep in range(10):
        model.train()
        idx = torch.randperm(len(Xtr_t))
        tot = 0.0
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            out = model(Xtr_t[b])
            loss = F.cross_entropy(out, ytr_t[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        log(f"epoch {ep+1}/10 loss {tot/max(1,(len(idx)+511)//512):.4f}")

    @torch.no_grad()
    def eval_split(X, y, name):
        model.eval()
        preds = []
        for s in range(0, len(X), 2048):
            preds.append(model(torch.tensor(X[s:s + 2048])).argmax(1).numpy())
        p = np.concatenate(preds)
        f1, per = macro_f1(y, p)
        res = {"split": name, "n": len(y), "macro_f1": round(f1, 4),
               "acc": round(float((p == y).mean()), 4), "per_class_f1": per,
               "true_counts": [int((y == c).sum()) for c in (0, 1, 2)]}
        log(json.dumps(res))
        return res

    res = {"val_image": eval_split(Xva, yva, "fold1_images"), "test_image": eval_split(Xte, yte, "yarisma_images")}
    out = ROOT / "results/image_level"; out.mkdir(parents=True, exist_ok=True)
    (out / "image_level_metrics.json").write_text(json.dumps(res, indent=2))
    log("saved results/image_level/image_level_metrics.json")


if __name__ == "__main__":
    main()
