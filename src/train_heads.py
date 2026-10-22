"""Train the 3 heads (BI-RADS / density / quadrant) on cached frozen features.

Balanced subset by default. Evaluates on the held-out fold and full YARISMA.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FEATS = {
    "gulluk": ROOT / "notebooks/out/features_full_gulluk_convnext_small.npz",
    "dinov3": ROOT / "notebooks/out/features_full_dinov3_vitb16.npz",
}
VIEW_ID = {"CC": 0, "MLO": 1}
SIDE_ID = {"L": 0, "R": 1}
ZONES = ["CENTER", "UOQ", "UIQ", "LOQ", "LIQ"]
DENS = {"A": 0, "B": 1, "C": 2, "D": 3}
B3 = {"0": 0, "1-2": 1, "4-5": 2}


def parse_zones(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return []
    return [z for z in str(v).split("|") if z in ZONES]


class Attn(nn.Module):
    def __init__(self, dim, hid=128):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(dim, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, x):
        a = self.w(x).squeeze(-1)
        w = torch.softmax(a, 0)
        return (x * w[:, None]).sum(0)


class HieraMIL3(nn.Module):
    def __init__(self, dim=768, hid=256):
        super().__init__()
        self.view_emb = nn.Embedding(2, dim)
        self.side_emb = nn.Embedding(2, dim)
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.att = Attn(hid)
        self.head_b = nn.Linear(hid, 3)
        self.head_d = nn.Linear(hid, 4)
        self.head_q = nn.Linear(hid, len(ZONES))

    def forward(self, x, vi, si):
        h = self.proj(x + self.view_emb(vi) + self.side_emb(si))
        sides = {}
        for s in (0, 1):
            m = si == s
            if m.any():
                sides[s] = self.att(h[m])
        st = torch.stack([sides[s] for s in sorted(sides)])
        z = self.att(st)
        q = {s: self.head_q(v) for s, v in sides.items()}
        return self.head_b(z), self.head_d(z), q


def load_view_meta():
    img = pd.read_csv(ROOT / "labels/images.csv")
    img["patient_id"] = img.patient_id.astype(str)
    img = img.sort_values(["shard", "row_in_shard"])
    meta = {
        pid: ([(VIEW_ID[v], SIDE_ID["L" if l == "left" else "R"]) for v, l in zip(g.view, g.laterality)])
        for pid, g in img.groupby("patient_id")
    }
    man = pd.read_csv(ROOT / "labels/yarisma_manifest.csv")
    man["patient_id"] = man.patient_id.astype(str)
    meta.update({
        pid: ([(VIEW_ID[v], SIDE_ID[l]) for v, l in zip(g.view, g.laterality)])
        for pid, g in man.groupby("patient_id", sort=False)
    })
    return meta


def make_items(keys, z, view_meta, cases):
    items = []
    for k in keys:
        pid = k.split("_", 1)[1]
        row = cases.loc[pid]
        arr = z[k]
        vs = view_meta[pid]
        if len(vs) != arr.shape[0]:
            continue
        qr = parse_zones(row.quadrant_right)
        ql = parse_zones(row.quadrant_left)
        items.append({
            "pid": pid,
            "feat": arr.astype(np.float32),
            "vi": np.array([v for v, _ in vs], dtype=np.int64),
            "si": np.array([s for _, s in vs], dtype=np.int64),
            "b3": B3[str(row.birads3)],
            "dens": DENS[str(row.density)],
            "q": {SIDE_ID["R"]: qr, SIDE_ID["L"]: ql},
        })
    return items


def stratified_keys(keys, z, cases, per_class, rng):
    by = {0: [], 1: [], 2: []}
    for k in keys:
        pid = k.split("_", 1)[1]
        by[B3[str(cases.loc[pid, "birads3"])]].append(k)
    chosen = []
    for c in (0, 1, 2):
        rng.shuffle(by[c])
        chosen += by[c][:per_class]
    return chosen


def to_tensors(it, device):
    return (torch.tensor(it["feat"], device=device),
            torch.tensor(it["vi"], device=device),
            torch.tensor(it["si"], device=device))


def quadrant_ce(logits_by_side, q, device):
    losses = []
    for s, zones in q.items():
        if s not in logits_by_side or not zones:
            continue
        y = torch.zeros(len(ZONES), device=device)
        for zn in zones:
            y[ZONES.index(zn)] = 1.0
        losses.append(F.binary_cross_entropy_with_logits(logits_by_side[s], y))
    return torch.stack(losses).mean() if losses else None


def train_epoch(model, items, opt, device, lam_aux=1.0):
    model.train()
    tot = 0.0
    for it in items:
        x, vi, si = to_tensors(it, device)
        lb, ld, lq = model(x, vi, si)
        yb = torch.tensor([it["b3"]], device=device)
        loss_b = F.cross_entropy(lb[None], yb)
        gate = torch.softmax(lb, 0)[it["b3"]].detach()
        loss_d = F.cross_entropy(ld[None], torch.tensor([it["dens"]], device=device))
        loss_q = quadrant_ce(lq, it["q"], device)
        loss = 0.70 * loss_b + lam_aux * gate * (0.15 * loss_d + (0.15 * loss_q if loss_q is not None else 0.0))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss.item())
    return tot / len(items)


@torch.no_grad()
def predict(model, items, device):
    model.eval()
    out = []
    for it in items:
        x, vi, si = to_tensors(it, device)
        lb, ld, lq = model(x, vi, si)
        out.append({
            "pid": it["pid"],
            "b3": int(lb.argmax()),
            "dens": int(ld.argmax()),
            "q": {s: int(v.argmax()) for s, v in lq.items()},
            "b3_prob": torch.softmax(lb, 0).cpu().numpy(),
        })
    return out


def f1_macro(y, p, k=3):
    f1s = []
    for c in range(k):
        tp = int(((np.array(p) == c) & (np.array(y) == c)).sum())
        fp = int(((np.array(p) == c) & (np.array(y) != c)).sum())
        fn = int(((np.array(p) != c) & (np.array(y) == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)), [round(x, 3) for x in f1s]


def evaluate(items, preds, name):
    y = [it["b3"] for it in items]
    p = [pr["b3"] for pr in preds]
    macro, per = f1_macro(y, p)
    acc = float(np.mean(np.array(y) == np.array(p)))
    dens_y = [it["dens"] for it in items]
    dens_p = [pr["dens"] for pr in preds]
    dens_macro, dens_per = f1_macro(dens_y, dens_p, 4)
    side_hits, side_tot = 0, 0
    for it, pr in zip(items, preds):
        for s, zones in it["q"].items():
            if zones and s in pr["q"]:
                side_tot += 1
                side_hits += int(ZONES[pr["q"][s]] in zones)
    q_acc = side_hits / max(side_tot, 1)
    scores = []
    for it, pr in zip(items, preds):
        if pr["b3"] != it["b3"]:
            scores.append(0.0)
            continue
        s = 0.70 + 0.15 * float(pr["dens"] == it["dens"])
        hits = [int(ZONES[pr["q"][s2]] in zones) for s2, zones in it["q"].items() if zones and s2 in pr["q"]]
        s += 0.15 * (float(np.mean(hits)) if hits else 1.0)
        scores.append(s)
    scores = np.array(scores)
    rng = np.random.default_rng(0)
    boots = np.array([scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(1000)])
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "split": name, "n": len(items),
        "birads_macro_f1": round(macro, 4), "birads_acc": round(acc, 4),
        "birads_per_class_f1": per,
        "density_macro_f1": round(dens_macro, 4), "density_acc": round(float(np.mean(np.array(dens_y) == np.array(dens_p))), 4),
        "density_per_class_f1": dens_per,
        "quadrant_side_acc": round(q_acc, 4), "quadrant_sides": side_tot,
        "gated_score": round(float(scores.mean()), 4), "gated_ci95": [round(ci[0], 4), round(ci[1], 4)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="gulluk", choices=list(FEATS))
    ap.add_argument("--features", default=None, help="feature npz path (overrides backbone default)")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    rng = np.random.default_rng(args.seed)

    z = np.load(args.features if args.features else FEATS[args.backbone])
    tr_keys = [k for k in z.keys() if k.startswith("tr_")]
    va_keys = [k for k in z.keys() if k.startswith("va_")]
    te_keys = [k for k in z.keys() if k.startswith("te_")]

    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str)
    cases = cases.set_index("patient_id")
    view_meta = load_view_meta()

    train_keys = stratified_keys(tr_keys, z, cases, args.per_class, rng)
    val_keys = stratified_keys(va_keys, z, cases, args.val_per_class, rng)
    train_items = make_items(train_keys, z, view_meta, cases)
    val_items = make_items(val_keys, z, view_meta, cases)
    test_items = make_items(te_keys, z, view_meta, cases)
    print(f"train {len(train_items)} | val {len(val_items)} | test {len(test_items)}", flush=True)

    dim = train_items[0]["feat"].shape[1]
    model = HieraMIL3(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        np.random.shuffle(train_items)
        loss = train_epoch(model, train_items, opt, device)
        print(f"epoch {ep+1}/{args.epochs} loss {loss:.4f}", flush=True)

    metrics = {
        "backbone": args.backbone,
        "train_cases": len(train_items), "val_cases": len(val_items), "test_cases": len(test_items),
        "epochs": args.epochs,
        "val": evaluate(val_items, predict(model, val_items, device), "val"),
        "test": evaluate(test_items, predict(model, test_items, device), "yarisma"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": dim, "args": vars(args)}, out / "heads_trackA.pt")
    (out / "track_a_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
