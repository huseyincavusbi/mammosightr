"""Batched version: ordinal density loss + multi-seed soup."""
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
from train_heads import SIDE_ID, VIEW_ID, ZONES, load_view_meta  # noqa: E402
from ensemble_calibrate import evaluate, fit_temperature, tune_bias  # noqa: E402

B3 = {"0": 0, "1-2": 1, "4-5": 2}
DENS = {"A": 0, "B": 1, "C": 2, "D": 3}
FEAT = Path(os.environ.get("FEATURES_FILE", ROOT / "notebooks/out/features_full_gulluk_convnext_small.npz"))
MAXV = 4


def log(*a):
    print(*a, flush=True)


def parse_zones(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return []
    return [z for z in str(v).split("|") if z in ZONES]


class Attn(nn.Module):
    def __init__(self, dim, hid=128):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, x, mask):
        a = self.w(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(a, -1)
        return (x * w.unsqueeze(-1)).sum(-2)


class HieraMILB(nn.Module):
    def __init__(self, dim=768, hid=256):
        super().__init__()
        self.view_emb = nn.Embedding(2, dim)
        self.side_emb = nn.Embedding(2, dim)
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.att = Attn(hid)
        self.head_b = nn.Linear(hid, 3)
        self.head_d = nn.Linear(hid, 4)
        self.head_q = nn.Linear(hid, len(ZONES))

    def forward(self, x, mask, vi, si):
        h = self.proj(x + self.view_emb(vi) + self.side_emb(si))
        sides, smasks = [], []
        for s in (0, 1):
            m = (si == s) & mask
            sides.append(self.att(h, m))
            smasks.append(m.any(1))
        st = torch.stack(sides, 1)
        sm = torch.stack(smasks, 1)
        z = self.att(st, sm)
        q = self.head_q(st)
        return self.head_b(z), self.head_d(z), q, sm


def make_batch(items, device):
    b = len(items)
    dim = items[0]["feat"].shape[1]
    x = np.zeros((b, MAXV, dim), np.float32)
    mask = np.zeros((b, MAXV), bool)
    vi = np.zeros((b, MAXV), np.int64)
    si = np.zeros((b, MAXV), np.int64)
    for i, it in enumerate(items):
        n = len(it["feat"])
        x[i, :n] = it["feat"]; mask[i, :n] = True
        vi[i, :n] = it["vi"]; si[i, :n] = it["si"]
    return (torch.tensor(x, device=device), torch.tensor(mask, device=device),
            torch.tensor(vi, device=device), torch.tensor(si, device=device))


def q_targets(items, device):
    t = torch.zeros(len(items), 2, len(ZONES), device=device)
    m = torch.zeros(len(items), 2, dtype=torch.bool, device=device)
    for i, it in enumerate(items):
        for s, zs in it["q"].items():
            if zs:
                m[i, s] = True
                for z in zs:
                    t[i, s, ZONES.index(z)] = 1.0
    return t, m


def emd_density(logits, y):
    p = torch.softmax(logits, -1)
    d = torch.arange(4, device=logits.device).float()
    return (p * (d[None, :] - y[:, None]).abs()).mean()


def batch_loss(lb, ld, lq, qm, batch, device, loss_type):
    yb = torch.tensor([it["b3"] for it in batch], device=device)
    yd = torch.tensor([it["dens"] for it in batch], device=device)
    qt, qmask = q_targets(batch, device)
    loss_b = F.cross_entropy(lb, yb)
    gate = torch.softmax(lb, 1)[torch.arange(len(batch)), yb].detach()
    loss_d = emd_density(ld, yd) if loss_type == "emd" else F.cross_entropy(ld, yd)
    lq_all = F.binary_cross_entropy_with_logits(lq, qt, reduction="none").mean(-1)
    loss_q = (lq_all * qmask).sum() / qmask.sum().clamp_min(1)
    return (0.70 * loss_b + gate * (0.15 * loss_d + 0.15 * loss_q)).mean()


def train(items, seed, loss_type, device, epochs=5, bs=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HieraMILB(items[0]["feat"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        idx = np.random.permutation(len(items))
        tot = 0.0
        for s in range(0, len(idx), bs):
            batch = [items[i] for i in idx[s:s + bs]]
            x, m, vi, si = make_batch(batch, device)
            lb, ld, lq, qm = model(x, m, vi, si)
            loss = batch_loss(lb, ld, lq, qm, batch, device, loss_type)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        log(f"  [{loss_type} seed {seed}] epoch {ep+1}/{epochs} loss {tot/max(1, (len(idx)+bs-1)//bs):.4f}")
    return model


@torch.no_grad()
def infer(model, items, device, bs=512):
    model.eval()
    out = {}
    for s in range(0, len(items), bs):
        batch = items[s:s + bs]
        x, m, vi, si = make_batch(batch, device)
        lb, ld, lq, qm = model(x, m, vi, si)
        lb, ld, lq = lb.cpu().numpy(), ld.cpu().numpy(), lq.cpu().numpy()
        for i, it in enumerate(batch):
            out[it["pid"]] = {"b": lb[i], "d": ld[i],
                              "q": {s2: lq[i, s2] for s2 in range(2) if qm[i, s2]}}
    return out


def soup(runs):
    out = {}
    for pid in runs[0]:
        out[pid] = {"b": np.mean([r[pid]["b"] for r in runs], 0),
                    "d": np.mean([r[pid]["d"] for r in runs], 0),
                    "q": {s: np.mean([r[pid]["q"][s] for r in runs], 0) for s in runs[0][pid]["q"]}}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    out = ROOT / "results/improve"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(FEAT)
    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str); cases = cases.set_index("patient_id")
    meta = load_view_meta()

    def build(keys):
        out = []
        for k in keys:
            pid = k.split("_", 1)[1]
            row = cases.loc[pid]; vs = meta[pid]
            out.append({"pid": pid, "feat": z[k].astype(np.float32),
                        "vi": np.array([v for v, _ in vs]), "si": np.array([s for _, s in vs]),
                        "b3": B3[str(row.birads3)], "dens": DENS[str(row.density)],
                        "q": {1: parse_zones(row.quadrant_right), 0: parse_zones(row.quadrant_left)}})
        return out

    tr = build([k for k in z.keys() if k.startswith("tr_")])
    va = build([k for k in z.keys() if k.startswith("va_")])
    te = build([k for k in z.keys() if k.startswith("te_")])
    log(f"train {len(tr)} val {len(va)} test {len(te)}")

    yv_b = np.array([it["b3"] for it in va]); yv_d = np.array([it["dens"] for it in va])
    results = {}
    for loss_type in ("ce", "emd"):
        log(f"=== density loss: {loss_type} ===")
        vrun, trun = [], []
        for seed in (0, 1, 2):
            m = train(tr, seed, loss_type, device)
            if loss_type == "ce":
                torch.save({"state": m.state_dict(), "dim": tr[0]["feat"].shape[1]},
                           out / f"soup_ce_seed{seed}.pt")
            vrun.append(infer(m, va, device)); trun.append(infer(m, te, device))
        vs, ts = soup(vrun), soup(trun)
        sb = np.stack([vs[p]["b"] for p in vs]); sd = np.stack([vs[p]["d"] for p in vs])
        tb = fit_temperature(sb, yv_b); td = fit_temperature(sd, yv_d)
        bb = tune_bias(sb, yv_b, 3); bd = tune_bias(sd, yv_d, 4)
        log(f"  calib T_b={tb:.2f} T_d={td:.2f} bias_b={bb.round(2).tolist()} bias_d={bd.round(2).tolist()}")
        base = evaluate(te, ts, device, name=f"soup_{loss_type}")
        cal = evaluate(te, ts, device, temp_b=tb, temp_d=td, bias_b=bb, bias_d=bd, name=f"soup_{loss_type}_cal")
        results[loss_type] = {"base": base, "calibrated": cal,
                              "temperature": [float(tb), float(td)], "bias_b": bb.tolist(), "bias_d": bd.tolist()}
        log(f"  {loss_type} base: " + json.dumps(base))
        log(f"  {loss_type} cal : " + json.dumps(cal))

    (out / "improve_metrics.json").write_text(json.dumps(results, indent=2))
    log("saved results/improve/improve_metrics.json")


if __name__ == "__main__":
    main()
