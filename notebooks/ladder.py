"""Frozen-feature backbone ladder for MammosightTR (Kaggle script kernel).

MODE="smoke": stratified 240 train + 240 val cases, 60 YARISMA test cases.
MODE="full":  patient-level fold 1 val, folds 2-4 train, full YARISMA test.

Data is indexed from all 23 parquet shards' metadata (cheap), then images are
read shard-by-shard so memory stays flat. Backbones are frozen; a hierarchical
attention MIL head (views -> side -> case) is trained on case-level BI-RADS.

Kaggle: attach the label/weights dataset and the YARISMA dataset; set
KAGGLE_LAB_DIR / KAGGLE_YAR_DIR if their mount paths differ from the defaults.
"""
MODE = "full"

import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

subprocess.run(f"{sys.executable} -m pip install -q -U transformers timm huggingface_hub opencv-python-headless",
               shell=True, check=False)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEV == "cuda"
torch.backends.cudnn.benchmark = True
print(f"device: {DEV} {torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}", flush=True)

IN_LAB = Path(os.environ.get("KAGGLE_LAB_DIR", "/kaggle/input/mammosightr-labels-convnext-ftune"))
IN_YAR = Path(os.environ.get("KAGGLE_YAR_DIR", "/kaggle/input/mammosightr-yarisma-768"))
OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)

PARAMS = {
    "smoke": dict(per_class_train=80, per_class_val=80, per_class_test=20, epochs=3, max_train=None),
    "full": dict(per_class_train=None, per_class_val=None, per_class_test=None, epochs=3, max_train=None),
}[MODE]
print(f"mode={MODE} params={PARAMS}", flush=True)

B3 = {0: 0, 1: 1, 2: 1, 4: 2, 5: 2}
B3_STR = {"0": 0, "1-2": 1, "4-5": 2}
DENS = {"A": 0, "B": 1, "C": 2, "D": 3}
VIEW_ID = {"CC": 0, "MLO": 1}
SIDE_ID = {"L": 0, "R": 1}
N_SHARDS = 23


def log(msg):
    print(msg, flush=True)


labels = pd.read_csv(IN_LAB / "cases.csv")
fold_df = pd.read_csv(IN_LAB / "folds.csv")
manifest = pd.read_csv(IN_YAR / "manifest.csv")
for _df in (labels, fold_df, manifest):
    _df["patient_id"] = _df.patient_id.astype(str)

log("downloading parquet shards from huggingface")
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

shard_paths = {}
for i in range(N_SHARDS):
    name = f"data/train-{i:05d}-of-00023.parquet"
    shard_paths[name] = hf_hub_download("gulluk/mammosightr-preprocessed", name, repo_type="dataset")
log(f"downloaded {len(shard_paths)} shards")

META_COLS = ["patient_id", "view", "laterality", "patient_birads", "breast_composition"]
meta_frames = []
for name, p in shard_paths.items():
    t = pq.read_table(p, columns=META_COLS)
    m = t.to_pandas()
    m["shard"] = name
    m["row"] = np.arange(len(m))
    meta_frames.append(m)
meta = pd.concat(meta_frames, ignore_index=True)
log(f"meta rows: {len(meta)} | cases: {meta.patient_id.nunique()}")

row_of = {pid: list(g.index) for pid, g in meta.groupby("patient_id")}
pid_label = {pid: B3[int(meta.loc[ix[0], "patient_birads"])] for pid, ix in row_of.items()}
rng = np.random.default_rng(0)


def stratified(pids, per_class, rng):
    out = []
    for c in (0, 1, 2):
        cand = [p for p in pids if pid_label.get(p) == c]
        rng.shuffle(cand)
        out += cand[: per_class if per_class else len(cand)]
    return out


if MODE == "smoke":
    all_pids = list(pid_label)
    rng.shuffle(all_pids)
    train_pids = set(stratified(all_pids, PARAMS["per_class_train"], rng))
    remaining = [p for p in all_pids if p not in train_pids]
    val_pids = set(stratified(remaining, PARAMS["per_class_val"], rng))
else:
    tr = set(fold_df[fold_df.fold != 1].patient_id)
    va = set(fold_df[fold_df.fold == 1].patient_id)
    train_pids = {p for p in pid_label if p in tr}
    val_pids = {p for p in pid_label if p in va}

test_meta = labels[labels.split == "test"].set_index("patient_id")
test_by_c = {0: [], 1: [], 2: []}
for pid, r in test_meta.iterrows():
    test_by_c[B3_STR[str(r["birads3"])]].append(pid)
test_pids = []
for c in (0, 1, 2):
    cand = test_by_c[c]
    rng.shuffle(cand)
    test_pids += cand[: PARAMS["per_class_test"] if PARAMS["per_class_test"] else len(cand)]
log(f"train cases: {len(train_pids)} | val cases: {len(val_pids)} | test cases: {len(test_pids)}")


def pq_items(pids):
    items = []
    for pid in pids:
        idxs = row_of[pid]
        views = []
        for i in idxs:
            r = meta.loc[i]
            views.append((("pq", r["shard"], int(r["row"])), VIEW_ID[r["view"]],
                          SIDE_ID["L" if r["laterality"] == "left" else "R"]))
        y = B3[int(meta.loc[idxs[0], "patient_birads"])]
        d = DENS[str(meta.loc[idxs[0], "breast_composition"])]
        items.append((pid, views, y, d))
    return items


def yar_items(pids):
    items = []
    for pid in pids:
        rows = manifest[manifest.patient_id == pid]
        views = [(("yar", pid, r.image_filename), VIEW_ID[r.view], SIDE_ID[r.laterality])
                 for r in rows.itertuples()]
        y = B3_STR[str(test_meta.loc[pid, "birads3"])]
        items.append((pid, views, y, None))
    return items


class GullukConvNeXt:
    name = "gulluk_convnext_small"

    def __init__(self):
        import timm
        m = timm.create_model("convnext_small", pretrained=False)
        m.head.fc = nn.Linear(768, 768)
        m.birads_layer = nn.Sequential(nn.Linear(768, 5))
        m.density_layer = nn.Sequential(nn.Linear(768, 4))
        sd = torch.load(IN_LAB / "convnext_small_ftune.pth", map_location="cpu", weights_only=False)
        m.load_state_dict(sd)
        self.m = m.eval().to(DEV)

    @torch.no_grad()
    def feats(self, imgs):
        xs = []
        for im in imgs:
            a = np.array(im)
            if a.shape != (1024, 768):
                import cv2
                a = cv2.resize(a, (768, 1024))
            x = a.astype(np.float32)
            x = x - x.min()
            x = x / max(float(x.max()), 1e-6)
            xs.append(np.stack([x, x, x]))
        x = torch.from_numpy(np.stack(xs)).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            f = self.m.forward_head(self.m.forward_features(x), pre_logits=True)
        return f.float().cpu().numpy()


class DinoV3:
    name = "dinov3_vitb16"

    def __init__(self):
        import timm
        self.m = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0,
                                   dynamic_img_size=True).eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(self, imgs):
        xs = []
        for im in imgs:
            a = np.array(im.convert("RGB"))
            if a.shape[:2] != (1024, 768):
                import cv2
                a = cv2.resize(a, (768, 1024))
            xs.append(a)
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().to(DEV) / 255.0
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            f = self.m.forward_head(self.m.forward_features(x), pre_logits=True)
        return f.float().cpu().numpy()


def extract_pq(bb, items):
    store = {pid: [None] * len(views) for pid, views, y, d in items}
    by_shard = {}
    for pid, views, y, d in items:
        for slot, (src, vi, si) in enumerate(views):
            by_shard.setdefault(src[1], []).append((pid, slot, src[2]))
    total = sum(len(v) for v in by_shard.values())
    done = 0
    t0 = time.time()
    for name, pairs in by_shard.items():
        t = pq.read_table(shard_paths[name], columns=["image"])
        col = t.column("image")
        by_case = {}
        for pid, slot, row in pairs:
            by_case.setdefault(pid, []).append((slot, row))
        for pid, sp in by_case.items():
            imgs = [Image.open(io.BytesIO(col[r].as_py()["bytes"])).convert("L") for _, r in sp]
            fs = bb.feats(imgs)
            for (slot, _), f in zip(sp, fs):
                store[pid][slot] = f
        del t, col
        done += len(pairs)
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else 0
        log(f"  {bb.name}: {done}/{total} views | {rate:.1f} view/s | elapsed {el/60:.1f} min | ETA {eta:.1f} min")
    return {pid: np.stack(v) for pid, v in store.items()}


def extract_yar(bb, items):
    feats = {}
    total = len(items)
    t0 = time.time()
    for i, (pid, views, y, d) in enumerate(items, 1):
        imgs = [Image.open(IN_YAR / src[1] / src[2]).convert("L") for src, vi, si in views]
        feats[pid] = bb.feats(imgs)
        if i % 10 == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el > 0 else 0
            log(f"  {bb.name} test: {i}/{total} cases | {rate:.2f} case/s")
    return feats


def standardize(tr, others):
    allf = np.concatenate(list(tr.values()), 0)
    mu = allf.mean(0)
    sd = allf.std(0) + 1e-6
    for d in [tr, *others]:
        for k in d:
            d[k] = ((d[k] - mu) / sd).astype(np.float32)


class Attn(nn.Module):
    def __init__(self, dim, hid=128):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, x, mask):
        a = self.w(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(a, 0)
        return (x * w[:, None]).sum(0)


class HieraMIL(nn.Module):
    def __init__(self, dim=768, hid=256):
        super().__init__()
        self.view_emb = nn.Embedding(2, dim)
        self.side_emb = nn.Embedding(2, dim)
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.att = Attn(hid)
        self.head = nn.Linear(hid, 3)
        self.dens = nn.Linear(hid, 4)

    def forward(self, x, vi, si):
        x = self.proj(x + self.view_emb(vi) + self.side_emb(si))
        sides = []
        for s in (0, 1):
            m = si == s
            if m.any():
                sides.append(self.att(x[m], torch.ones(int(m.sum()), dtype=torch.bool, device=x.device)))
        if not sides:
            z = x.mean(0)
        else:
            st = torch.stack(sides)
            z = self.att(st, torch.ones(len(st), dtype=torch.bool, device=x.device))
        return self.head(z), self.dens(z)


def tensorize(feats, items):
    data = {}
    for pid, views, y, d in items:
        vi = torch.tensor([v[1] for v in views], dtype=torch.long, device=DEV)
        si = torch.tensor([v[2] for v in views], dtype=torch.long, device=DEV)
        data[pid] = (torch.tensor(feats[pid], dtype=torch.float32, device=DEV), vi, si, y, d)
    return data


def train_head(feats, items, epochs):
    dim = next(iter(feats.values())).shape[1]
    model = HieraMIL(dim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    data = tensorize(feats, items)
    keys = list(data)
    ys = np.array([data[k][3] for k in keys])
    cnt = np.bincount(ys, minlength=3).clip(1)
    w = torch.tensor(len(ys) / (3 * cnt), dtype=torch.float32, device=DEV)
    for ep in range(epochs):
        model.train()
        np.random.shuffle(keys)
        tot = 0.0
        for pid in keys:
            x, vi, si, y, d = data[pid]
            b, dd = model(x, vi, si)
            loss = F.cross_entropy(b[None], torch.tensor([y], device=DEV), weight=w)
            if d is not None:
                loss = loss + 0.2 * F.cross_entropy(dd[None], torch.tensor([d], device=DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        log(f"    epoch {ep+1}/{epochs} loss {tot/len(keys):.4f}")
    return model


@torch.no_grad()
def predict(model, feats, items):
    model.eval()
    data = tensorize(feats, items)
    preds = {}
    for pid, (x, vi, si, y, d) in data.items():
        b, _ = model(x, vi, si)
        preds[pid] = int(b.argmax())
    return preds


def f1_report(items, preds):
    ys = np.array([it[2] for it in items])
    ps = np.array([preds[it[0]] for it in items])
    f1s, per = [], {}
    for c in (0, 1, 2):
        tp = int(((ps == c) & (ys == c)).sum())
        fp = int(((ps == c) & (ys != c)).sum())
        fn = int(((ps != c) & (ys == c)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        f1s.append(f1)
        per[int(c)] = round(f1, 3)
    acc = float((ps == ys).mean())
    conf = [[int(((ys == t) & (ps == p)).sum()) for p in (0, 1, 2)] for t in (0, 1, 2)]
    return {"macro_f1": round(float(np.mean(f1s)), 4), "acc": round(acc, 4),
            "per_class_f1": per, "true_counts": [int((ys == c).sum()) for c in (0, 1, 2)], "confusion": conf}


train_items_ = pq_items(train_pids)
val_items_ = pq_items(val_pids)
test_items_ = yar_items(test_pids)
log(f"items: train {len(train_items_)} | val {len(val_items_)} | test {len(test_items_)}")

BACKBONES = [GullukConvNeXt, DinoV3]
results = {}
for cls in BACKBONES:
    log(f"\n===== {cls.name} =====")
    try:
        t0 = time.time()
        bb = cls()
        log(f"  loaded in {time.time()-t0:.0f}s")
        tr_f = extract_pq(bb, train_items_)
        va_f = extract_pq(bb, val_items_)
        te_f = extract_yar(bb, test_items_)
        standardize(tr_f, [va_f, te_f])
        model = train_head(tr_f, train_items_, PARAMS["epochs"])
        vp = predict(model, va_f, val_items_)
        tp = predict(model, te_f, test_items_)
        res = {"val": f1_report(val_items_, vp), "test": f1_report(test_items_, tp)}
        results[cls.name] = res
        log(f"  RESULT {cls.name}: {json.dumps(res)}")
        np.savez_compressed(OUT / f"features_{MODE}_{cls.name}.npz",
                            **{f"tr_{k}": v for k, v in tr_f.items()},
                            **{f"va_{k}": v for k, v in va_f.items()},
                            **{f"te_{k}": v for k, v in te_f.items()})
    except Exception as e:
        results[cls.name] = {"error": f"{type(e).__name__}: {e}"}
        log(f"  FAILED {cls.name}: {type(e).__name__}: {e}")
        traceback.print_exc()

log("\n===== SUMMARY =====")
for k, v in results.items():
    log(f"{k}: {json.dumps(v)}")
(OUT / f"ladder_{MODE}_results.json").write_text(json.dumps({"mode": MODE, "results": results}, indent=2))
log("done")
