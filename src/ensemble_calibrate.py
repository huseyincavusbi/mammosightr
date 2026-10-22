"""Ensemble Gulluk + DINOv3 cached features, train 3 heads, calibrate, evaluate."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_heads import HieraMIL3, SIDE_ID, VIEW_ID, ZONES  # noqa: E402

B3 = {"0": 0, "1-2": 1, "4-5": 2}
DENS = {"A": 0, "B": 1, "C": 2, "D": 3}
FEAT_G = ROOT / "notebooks/out/features_full_gulluk_convnext_small.npz"
FEAT_D = ROOT / "notebooks/out/features_full_dinov3_vitb16.npz"


def log(*a):
    print(*a, flush=True)


def parse_zones(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return []
    return [z for z in str(v).split("|") if z in ZONES]


def load_meta():
    img = pd.read_csv(ROOT / "labels/images.csv")
    img["patient_id"] = img.patient_id.astype(str)
    img = img.sort_values(["shard", "row_in_shard"])
    meta = {pid: [(VIEW_ID[v], SIDE_ID["L" if l == "left" else "R"]) for v, l in zip(g.view, g.laterality)]
            for pid, g in img.groupby("patient_id")}
    man = pd.read_csv(ROOT / "labels/yarisma_manifest.csv")
    man["patient_id"] = man.patient_id.astype(str)
    meta.update({pid: [(VIEW_ID[v], SIDE_ID[l]) for v, l in zip(g.view, g.laterality)]
                 for pid, g in man.groupby("patient_id", sort=False)})
    return meta


def build_items(keys, zg, zd, meta, cases):
    items = []
    for k in keys:
        pid = k.split("_", 1)[1]
        row = cases.loc[pid]
        f = np.concatenate([zg[k], zd[k]], axis=1).astype(np.float32)
        vs = meta[pid]
        items.append({
            "pid": pid, "feat": f,
            "vi": np.array([v for v, _ in vs]), "si": np.array([s for _, s in vs]),
            "b3": B3[str(row.birads3)], "dens": DENS[str(row.density)],
            "q": {1: parse_zones(row.quadrant_right), 0: parse_zones(row.quadrant_left)},
        })
    return items


def gated_loss(lb, ld, lq, it, device):
    loss_b = F.cross_entropy(lb[None], torch.tensor([it["b3"]], device=device))
    gate = torch.softmax(lb, 0)[it["b3"]].detach()
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


def train(model, items, epochs, device):
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
            loss = gated_loss(lb, ld, lq, it, device)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        log(f"epoch {ep+1}/{epochs} loss {tot/len(order):.4f}")


@torch.no_grad()
def infer(model, items, device):
    out = {}
    for it in items:
        x = torch.tensor(it["feat"], device=device)
        vi = torch.tensor(it["vi"], device=device)
        si = torch.tensor(it["si"], device=device)
        lb, ld, lq = model(x, vi, si)
        out[it["pid"]] = {
            "b": lb.cpu().numpy(), "d": ld.cpu().numpy(),
            "q": {s: v.cpu().numpy() for s, v in lq.items()},
        }
    return out


def macro_f1(y, p, k=3):
    f1 = []
    for c in range(k):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1)), [round(x, 4) for x in f1]


def evaluate(items, pred, device, temp_b=1.0, temp_d=1.0, bias_b=None, bias_d=None, name="test"):
    bias_b = np.zeros(3) if bias_b is None else bias_b
    bias_d = np.zeros(4) if bias_d is None else bias_d
    yb, pb, yd, pdd, hits, sides, scores = [], [], [], [], 0, 0, []
    for it in items:
        o = pred[it["pid"]]
        b = int((o["b"] / temp_b + bias_b).argmax())
        d = int((o["d"] / temp_d + bias_d).argmax())
        yb.append(it["b3"]); pb.append(b); yd.append(it["dens"]); pdd.append(d)
        qpred = {s: int(v.argmax()) for s, v in o["q"].items()}
        for s, zs in it["q"].items():
            if zs and s in qpred:
                sides += 1
                hits += int(ZONES[qpred[s]] in zs)
        if b == it["b3"]:
            sc = 0.70 + 0.15 * float(d == it["dens"])
            hh = [int(ZONES[qpred[s]] in zs) for s, zs in it["q"].items() if zs and s in qpred]
            sc += 0.15 * (float(np.mean(hh)) if hh else 1.0)
            scores.append(sc)
        else:
            scores.append(0.0)
    yb, pb, yd, pdd = np.array(yb), np.array(pb), np.array(yd), np.array(pdd)
    bf1, bper = macro_f1(yb, pb, 3); df1, dper = macro_f1(yd, pdd, 4)
    scores = np.array(scores)
    rng = np.random.default_rng(0)
    boots = np.array([scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(1000)])
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    return {"split": name, "n": len(items), "birads_macro_f1": round(bf1, 4),
            "birads_acc": round(float((yb == pb).mean()), 4), "birads_per_class_f1": bper,
            "density_macro_f1": round(df1, 4), "density_acc": round(float((yd == pdd).mean()), 4),
            "density_per_class_f1": dper, "quadrant_side_acc": round(hits / max(sides, 1), 4),
            "quadrant_sides": sides, "gated_score": round(float(scores.mean()), 4),
            "gated_ci95": [round(ci[0], 4), round(ci[1], 4)]}


def fit_temperature(logits, y):
    best = (1e9, 1.0)
    for T in np.arange(0.5, 3.01, 0.05):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        nll = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
        if nll < best[0]:
            best = (nll, T)
    return best[1]


def tune_bias(logits, y, k, rounds=3, grid=np.arange(-1.0, 1.01, 0.1)):
    b = np.zeros(k)
    for _ in range(rounds):
        for c in range(k):
            best = (-1, b[c])
            for v in grid:
                bb = b.copy(); bb[c] = v
                f1, _ = macro_f1(np.array(y), (logits + bb).argmax(1), k)
                if f1 > best[0]:
                    best = (f1, v)
            b[c] = best[1]
    return b


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device: {device}")
    zg = np.load(FEAT_G); zd = np.load(FEAT_D)
    cases = pd.read_csv(ROOT / "labels/cases.csv"); cases["patient_id"] = cases.patient_id.astype(str); cases = cases.set_index("patient_id")
    meta = load_meta()

    tr = build_items([k for k in zg.keys() if k.startswith("tr_")], zg, zd, meta, cases)
    va = build_items([k for k in zg.keys() if k.startswith("va_")], zg, zd, meta, cases)
    te = build_items([k for k in zg.keys() if k.startswith("te_")], zg, zd, meta, cases)
    log(f"train {len(tr)} val {len(va)} test {len(te)} | dim {tr[0]['feat'].shape[1]}")

    model = HieraMIL3(tr[0]["feat"].shape[1]).to(device)
    train(model, tr, 5, device)
    pv = infer(model, va, device)
    pt = infer(model, te, device)

    base = evaluate(te, pt, device, name="ensemble_test")
    log("base: " + json.dumps(base))

    yv_b = np.array([it["b3"] for it in va])
    yv_d = np.array([it["dens"] for it in va])
    lb = np.stack([pv[it["pid"]]["b"] for it in va])
    ld = np.stack([pv[it["pid"]]["d"] for it in va])
    tb = fit_temperature(lb, yv_b)
    td = fit_temperature(ld, yv_d)
    bb = tune_bias(lb, yv_b, 3)
    bd = tune_bias(ld, yv_d, 4)
    log(f"temperature b={tb:.2f} d={td:.2f} | bias b={bb.round(2).tolist()} d={bd.round(2).tolist()}")

    cal = evaluate(te, pt, device, temp_b=tb, temp_d=td, bias_b=bb, bias_d=bd, name="ensemble_calibrated")
    log("calibrated: " + json.dumps(cal))
    val_cal = evaluate(va, pv, device, temp_b=tb, temp_d=td, bias_b=bb, bias_d=bd, name="val_calibrated")

    out = ROOT / "results/ensemble"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": tr[0]["feat"].shape[1]}, out / "ensemble_heads.pt")
    res = {"temperature": {"birads": float(tb), "density": float(td)},
           "bias": {"birads": bb.tolist(), "density": bd.tolist()},
           "val_calibrated": val_cal, "base_test": base, "calibrated_test": cal}
    (out / "ensemble_metrics.json").write_text(json.dumps(res, indent=2))
    log("saved results/ensemble/ensemble_metrics.json")


if __name__ == "__main__":
    main()
