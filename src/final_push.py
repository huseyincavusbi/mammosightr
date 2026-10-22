"""Final head-level push: 5-seed soup, density class weights, gated-objective
decision-rule tuning (birads/density biases + BI-RADS-0 abstention)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("MAMMO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from improve_batched import (B3, DENS, FEAT, HieraMILB, infer, make_batch,  # noqa: E402
                             parse_zones, q_targets, soup)
from train_heads import ZONES, load_view_meta  # noqa: E402
from ensemble_calibrate import evaluate  # noqa: E402


def log(*a):
    print(*a, flush=True)


def batch_loss_w(lb, ld, lq, qm, batch, device, wb, wd):
    yb = torch.tensor([it["b3"] for it in batch], device=device)
    yd = torch.tensor([it["dens"] for it in batch], device=device)
    qt, qmask = q_targets(batch, device)
    loss_b = F.cross_entropy(lb, yb, weight=wb)
    gate = torch.softmax(lb, 1)[torch.arange(len(batch)), yb].detach()
    loss_d = F.cross_entropy(ld, yd, weight=wd)
    lq_all = F.binary_cross_entropy_with_logits(lq, qt, reduction="none").mean(-1)
    loss_q = (lq_all * qmask).sum() / qmask.sum().clamp_min(1)
    return (0.70 * loss_b + gate * (0.15 * loss_d + 0.15 * loss_q)).mean()


def train_w(items, seed, device, wb, wd, epochs=5, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HieraMILB(items[0]["feat"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        idx = np.random.permutation(len(items)); tot = 0.0
        for s in range(0, len(idx), bs):
            batch = [items[i] for i in idx[s:s + bs]]
            x, m, vi, si = make_batch(batch, device)
            lb, ld, lq, qm = model(x, m, vi, si)
            loss = batch_loss_w(lb, ld, lq, qm, batch, device, wb, wd)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item())
        log(f"  [seed {seed}] epoch {ep+1}/{epochs} loss {tot/max(1,(len(idx)+bs-1)//bs):.4f}")
    return model


def fast_gated(items, pred, bias_b, bias_d, tau=0.0):
    scores = []
    for it in items:
        o = pred[it["pid"]]
        b = int((o["b"] + bias_b).argmax())
        d = int((o["d"] + bias_d).argmax())
        if tau > 0 and b != 0:
            p = torch.softmax(torch.tensor(o["b"] + bias_b), 0).numpy()
            if float(p.max()) < tau:
                b = 0
        qpred = {s: int(v.argmax()) for s, v in o["q"].items()}
        if b == it["b3"]:
            sc = 0.70 + 0.15 * float(d == it["dens"])
            hh = [int(ZONES[qpred[s]] in zs) for s, zs in it["q"].items() if zs and s in qpred]
            sc += 0.15 * (float(np.mean(hh)) if hh else 1.0)
            scores.append(sc)
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def tune(items, pred, grid=np.arange(-1.0, 1.01, 0.1), rounds=2):
    bb = np.zeros(3); bd = np.zeros(4)
    best = fast_gated(items, pred, bb, bd)
    for _ in range(rounds):
        for c in range(3):
            for v in grid:
                cand = bb.copy(); cand[c] = v
                s = fast_gated(items, pred, cand, bd)
                if s > best:
                    best, bb = s, cand
        for c in range(4):
            for v in grid:
                cand = bd.copy(); cand[c] = v
                s = fast_gated(items, pred, bb, cand)
                if s > best:
                    best, bd = s, cand
    tau, best_tau = 0.0, fast_gated(items, pred, bb, bd)
    for t in np.arange(0.3, 0.91, 0.05):
        s = fast_gated(items, pred, bb, bd, tau=float(t))
        if s > best_tau:
            best_tau, tau = s, float(t)
    return bb, bd, tau, best_tau


def apply_tau(pred, tau, bias_b):
    if tau <= 0:
        return pred
    out = {}
    for pid, o in pred.items():
        z = o["b"] + bias_b
        p = torch.softmax(torch.tensor(z), 0).numpy()
        b = o["b"]
        if int(p.argmax()) != 0 and float(p.max()) < tau:
            b = np.array([20.0, -20.0, -20.0], dtype=o["b"].dtype)
        out[pid] = {"b": b, "d": o["d"], "q": o["q"]}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    z = np.load(FEAT)
    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str); cases = cases.set_index("patient_id")
    meta = load_view_meta()

    def build(keys):
        out = []
        for k in keys:
            pid = k.split("_", 1)[1]; row = cases.loc[pid]; vs = meta[pid]
            out.append({"pid": pid, "feat": z[k].astype(np.float32),
                        "vi": np.array([v for v, _ in vs]), "si": np.array([s for _, s in vs]),
                        "b3": B3[str(row.birads3)], "dens": DENS[str(row.density)],
                        "q": {1: parse_zones(row.quadrant_right), 0: parse_zones(row.quadrant_left)}})
        return out

    tr = build([k for k in z.keys() if k.startswith("tr_")])
    va = build([k for k in z.keys() if k.startswith("va_")])
    te = build([k for k in z.keys() if k.startswith("te_")])
    log(f"train {len(tr)} val {len(va)} test {len(te)}")

    ytr_b = np.array([it["b3"] for it in tr]); ytr_d = np.array([it["dens"] for it in tr])
    wb = torch.tensor(len(ytr_b) / (3 * np.bincount(ytr_b, minlength=3).clip(1)), dtype=torch.float32, device=device)
    wd = torch.tensor(len(ytr_d) / (4 * np.bincount(ytr_d, minlength=4).clip(1)), dtype=torch.float32, device=device)
    log(f"class weights b={wb.cpu().numpy().round(2).tolist()} d={wd.cpu().numpy().round(2).tolist()}")

    runs_v, runs_t = [], []
    for seed in range(5):
        m = train_w(tr, seed, device, wb, wd)
        runs_v.append(infer(m, va, device)); runs_t.append(infer(m, te, device))
        torch.save({"state": m.state_dict(), "dim": tr[0]["feat"].shape[1]},
                   ROOT / f"results/improve/push_seed{seed}.pt")
    vs, ts = soup(runs_v), soup(runs_t)

    base_val = fast_gated(va, vs, np.zeros(3), np.zeros(4))
    bb, bd, tau, tuned_val = tune(va, vs)
    log(f"val: base {base_val:.4f} -> tuned {tuned_val:.4f} | bias_b={bb.round(2).tolist()} bias_d={bd.round(2).tolist()} tau={tau:.2f}")

    base_test = evaluate(te, ts, device, name="push_base")
    tuned_test = evaluate(te, apply_tau(ts, tau, bb), device, bias_b=bb, bias_d=bd, name="push_tuned")
    log("base: " + json.dumps(base_test))
    log("tuned: " + json.dumps(tuned_test))

    out = ROOT / "results/push"; out.mkdir(parents=True, exist_ok=True)
    (out / "push_metrics.json").write_text(json.dumps(
        {"class_weights": {"birads": wb.cpu().numpy().tolist(), "density": wd.cpu().numpy().tolist()},
         "val_base": base_val, "val_tuned": tuned_val, "bias_b": bb.tolist(), "bias_d": bd.tolist(), "tau": tau,
         "base_test": base_test, "tuned_test": tuned_test}, indent=2))
    log("saved results/push/push_metrics.json")


if __name__ == "__main__":
    main()
