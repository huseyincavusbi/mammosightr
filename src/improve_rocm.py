"""Ordinal density loss + multi-seed head soup + calibration."""
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
from train_heads import HieraMIL3, SIDE_ID, VIEW_ID, ZONES, load_view_meta  # noqa: E402
from ensemble_calibrate import evaluate, fit_temperature, macro_f1, tune_bias  # noqa: E402

B3 = {"0": 0, "1-2": 1, "4-5": 2}
DENS = {"A": 0, "B": 1, "C": 2, "D": 3}
FEAT = ROOT / "features/features_full_gulluk_convnext_small.npz"


def log(*a):
    print(*a, flush=True)


def parse_zones(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return []
    return [z for z in str(v).split("|") if z in ZONES]


def build_items(keys, z, meta, cases):
    out = []
    for k in keys:
        pid = k.split("_", 1)[1]
        row = cases.loc[pid]
        vs = meta[pid]
        out.append({
            "pid": pid, "feat": z[k].astype(np.float32),
            "vi": np.array([v for v, _ in vs]), "si": np.array([s for _, s in vs]),
            "b3": B3[str(row.birads3)], "dens": DENS[str(row.density)],
            "q": {1: parse_zones(row.quadrant_right), 0: parse_zones(row.quadrant_left)},
        })
    return out


def emd_density(logits, y):
    p = torch.softmax(logits, 0)
    d = torch.arange(4, device=logits.device).float()
    return (p * (d - y).abs()).sum()


def gated_loss(lb, ld, lq, it, device, loss_type):
    loss_b = F.cross_entropy(lb[None], torch.tensor([it["b3"]], device=device))
    gate = torch.softmax(lb, 0)[it["b3"]].detach()
    if loss_type == "emd":
        loss_d = emd_density(ld, torch.tensor(it["dens"], device=device))
    else:
        loss_d = F.cross_entropy(ld[None], torch.tensor([it["dens"]], device=device))
    ql = []
    for s, zs in it["q"].items():
        if s in lq and zs:
            y = torch.zeros(len(ZONES), device=device)
            for z in zs:
                y[ZONES.index(z)] = 1.0
            ql.append(F.binary_cross_entropy_with_logits(lq[s], y))
    loss_q = torch.stack(ql).mean() if ql else torch.tensor(0.0, device=device)
    return 0.70 * loss_b + gate * (0.15 * loss_d + 0.15 * loss_q)


def train(items, seed, loss_type, device, epochs=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HieraMIL3(items[0]["feat"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = items[:]
    for ep in range(epochs):
        np.random.shuffle(order)
        tot = 0.0
        for it in order:
            x = torch.tensor(it["feat"], device=device)
            vi = torch.tensor(it["vi"], device=device)
            si = torch.tensor(it["si"], device=device)
            lb, ld, lq = model(x, vi, si)
            loss = gated_loss(lb, ld, lq, it, device, loss_type)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        log(f"  [{loss_type} seed {seed}] epoch {ep+1}/{epochs} loss {tot/len(order):.4f}")
    return model


@torch.no_grad()
def infer(model, items, device):
    model.eval()
    out = {}
    for it in items:
        x = torch.tensor(it["feat"], device=device)
        vi = torch.tensor(it["vi"], device=device)
        si = torch.tensor(it["si"], device=device)
        lb, ld, lq = model(x, vi, si)
        out[it["pid"]] = {"b": lb.cpu().numpy(), "d": ld.cpu().numpy(),
                          "q": {s: v.cpu().numpy() for s, v in lq.items()}}
    return out


def soup(runs):
    keys = list(runs[0])
    out = {}
    for pid in keys:
        out[pid] = {
            "b": np.mean([r[pid]["b"] for r in runs], 0),
            "d": np.mean([r[pid]["d"] for r in runs], 0),
            "q": {s: np.mean([r[pid]["q"][s] for r in runs], 0) for s in runs[0][pid]["q"]},
        }
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    z = np.load(FEAT)
    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str); cases = cases.set_index("patient_id")
    meta = load_view_meta()

    tr = build_items([k for k in z.keys() if k.startswith("tr_")], z, meta, cases)
    va = build_items([k for k in z.keys() if k.startswith("va_")], z, meta, cases)
    te = build_items([k for k in z.keys() if k.startswith("te_")], z, meta, cases)
    log(f"train {len(tr)} val {len(va)} test {len(te)}")

    yv_b = np.array([it["b3"] for it in va]); yv_d = np.array([it["dens"] for it in va])
    results = {}
    for loss_type in ("ce", "emd"):
        log(f"=== density loss: {loss_type} ===")
        vrun, trun = [], []
        for seed in (0, 1, 2):
            m = train(tr, seed, loss_type, device)
            vrun.append(infer(m, va, device))
            trun.append(infer(m, te, device))
        vs, ts = soup(vrun), soup(trun)
        sb = np.stack([vs[p]["b"] for p in vs]); sd = np.stack([vs[p]["d"] for p in vs])
        tb = fit_temperature(sb, yv_b); td = fit_temperature(sd, yv_d)
        bb = tune_bias(sb, yv_b, 3); bd = tune_bias(sd, yv_d, 4)
        log(f"  calib: T_b={tb:.2f} T_d={td:.2f} bias_b={bb.round(2).tolist()} bias_d={bd.round(2).tolist()}")
        base = evaluate(te, ts, device, name=f"soup_{loss_type}")
        cal = evaluate(te, ts, device, temp_b=tb, temp_d=td, bias_b=bb, bias_d=bd, name=f"soup_{loss_type}_cal")
        results[loss_type] = {"base": base, "calibrated": cal,
                              "temperature": [float(tb), float(td)],
                              "bias_b": bb.tolist(), "bias_d": bd.tolist()}
        log(f"  {loss_type} base: " + json.dumps(base))
        log(f"  {loss_type} cal : " + json.dumps(cal))

    out = ROOT / "results/improve"; out.mkdir(parents=True, exist_ok=True)
    (out / "improve_metrics.json").write_text(json.dumps(results, indent=2))
    log("saved results/improve/improve_metrics.json")


if __name__ == "__main__":
    main()
