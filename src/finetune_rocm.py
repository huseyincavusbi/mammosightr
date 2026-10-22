"""Fine-tune Gulluk ConvNeXt-s (last stage) + HieraMIL 3 heads with ROCm.

Uses the exact Track-A balanced split. Caches the subset images as a memmap,
trains 3 epochs, evaluates fold-1 val each epoch and full YARISMA at the end.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

ROOT = Path(os.environ.get("MAMMO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from train_heads import HieraMIL3, SIDE_ID, VIEW_ID, ZONES  # noqa: E402

B3 = {"0": 0, "1-2": 1, "4-5": 2}
DENS = {"A": 0, "B": 1, "C": 2, "D": 3}


def log(*a):
    print(*a, flush=True)


def side_id(l):
    return SIDE_ID["L" if l == "left" else "R"]


def parse_zones(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return []
    return [z for z in str(v).split("|") if z in ZONES]


def build_cache(split_csv, npy_path, man_path, subset="train"):
    if Path(npy_path).exists() and Path(man_path).exists():
        return
    split = pd.read_csv(split_csv)
    pids = set(split[split.split == subset].patient_id.astype(str))
    img = pd.read_csv(ROOT / "labels/images.csv")
    img["patient_id"] = img.patient_id.astype(str)
    img = img[img.patient_id.isin(pids)].sort_values(["shard", "row_in_shard"])
    arrs, man = [], []
    for shard, g in img.groupby("shard", sort=True):
        col = pq.read_table(ROOT / "data/parquet" / shard, columns=["image"]).column("image")
        for r in g.itertuples():
            a = np.array(Image.open(io.BytesIO(col[int(r.row_in_shard)].as_py()["bytes"])).convert("L"))
            if a.shape != (1024, 768):
                a = cv2.resize(a, (768, 1024))
            arrs.append(a)
            man.append((r.patient_id, r.view, r.laterality))
        log(f"  cached {shard}: {len(man)}")
    np.save(npy_path, np.stack(arrs))
    pd.DataFrame(man, columns=["patient_id", "view", "laterality"]).to_csv(man_path, index=False)
    log(f"saved {npy_path} ({len(arrs)} images)")


def batch_images(arr, idxs, device):
    xs = []
    for i in idxs:
        a = np.asarray(arr[i], dtype=np.float32)
        a = a - a.min()
        a = a / max(float(a.max()), 1e-6)
        xs.append(np.stack([a, a, a]))
    return torch.from_numpy(np.stack(xs)).to(device)


def meta_for(pid, man_by_pid):
    g = man_by_pid[pid]
    vi = torch.tensor([VIEW_ID[v] for v in g.view], dtype=torch.long)
    si = torch.tensor([side_id(l) for l in g.laterality], dtype=torch.long)
    return vi, si


def gated_loss(lb, ld, lq, it, device):
    loss_b = F.cross_entropy(lb[None], torch.tensor([it["b3"]], device=device))
    gate = torch.softmax(lb, 0)[it["b3"]].detach()
    loss_d = F.cross_entropy(ld[None], torch.tensor([it["dens"]], device=device))
    qlosses = []
    for s, zones in it["q"].items():
        if s in lq and zones:
            y = torch.zeros(len(ZONES), device=device)
            for z in zones:
                y[ZONES.index(z)] = 1.0
            qlosses.append(F.binary_cross_entropy_with_logits(lq[s], y))
    loss_q = torch.stack(qlosses).mean() if qlosses else torch.tensor(0.0, device=device)
    return 0.70 * loss_b + gate * (0.15 * loss_d + 0.15 * loss_q)


@torch.no_grad()
def evaluate(pids, arr, man_by_pid, cases, backbone, head, stats, device, name):
    backbone.eval(); head.eval()
    yb, pb, yd, pd_, hits, sides, scores = [], [], [], [], 0, 0, []
    for pid in pids:
        idxs = [int(i) for i in man_by_pid[pid].index]
        x = batch_images(arr, idxs, device)
        with torch.autocast("cuda", dtype=torch.float16):
            f = backbone.forward_head(backbone.forward_features(x), pre_logits=True)
        f = ((f.float().cpu().numpy() - stats["mu"]) / stats["sd"])
        f = torch.tensor(f, dtype=torch.float32, device=device)
        vi, si = meta_for(pid, man_by_pid)
        lb, ld, lq = head(f, vi.to(device), si.to(device))
        row = cases.loc[pid]
        b = int(lb.argmax()); d = int(ld.argmax())
        yb.append(B3[str(row.birads3)])
        pb.append(b)
        yd.append(DENS[str(row.density)])
        pd_.append(d)
        qpred = {s: int(v.argmax()) for s, v in lq.items()}
        for s, key in ((1, "quadrant_right"), (0, "quadrant_left")):
            zs = parse_zones(row[key])
            if zs and s in qpred:
                sides += 1
                hits += int(ZONES[qpred[s]] in zs)
        if b == B3[str(row.birads3)]:
            sc = 0.70 + 0.15 * float(d == DENS[str(row.density)])
            zs2 = [(s, parse_zones(row[k])) for s, k in ((1, "quadrant_right"), (0, "quadrant_left"))]
            hh = [int(ZONES[qpred[s]] in z) for s, z in zs2 if z and s in qpred]
            sc += 0.15 * (float(np.mean(hh)) if hh else 1.0)
            scores.append(sc)
        else:
            scores.append(0.0)
    yb, pb, yd, pd_ = np.array(yb), np.array(pb), np.array(yd), np.array(pd_)
    def f1(y, p, k):
        out = []
        for c in range(k):
            tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
            out.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(out)), [round(x, 3) for x in out]
    bf1, bper = f1(yb, pb, 3)
    df1, dper = f1(yd, pd_, 4)
    scores = np.array(scores)
    rng = np.random.default_rng(0)
    boots = np.array([scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(1000)])
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    return {"split": name, "n": len(pids), "birads_macro_f1": round(bf1, 4), "birads_acc": round(float((yb == pb).mean()), 4),
            "birads_per_class_f1": bper, "density_macro_f1": round(df1, 4), "density_acc": round(float((yd == pd_).mean()), 4),
            "density_per_class_f1": dper, "quadrant_side_acc": round(hits / max(sides, 1), 4), "quadrant_sides": sides,
            "gated_score": round(float(scores.mean()), 4), "gated_ci95": [round(ci[0], 4), round(ci[1], 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-cases", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    split_csv = ROOT / "results/trackA_split.csv"
    log("building image caches")
    build_cache(split_csv, ROOT / "results/train_u8.npy", ROOT / "results/train_manifest.csv", "train")
    build_cache(split_csv, ROOT / "results/val_u8.npy", ROOT / "results/val_manifest.csv", "val")

    arr = np.load(ROOT / "results/train_u8.npy", mmap_mode="r")
    man = pd.read_csv(ROOT / "results/train_manifest.csv")
    man["patient_id"] = man.patient_id.astype(str)
    man["idx"] = np.arange(len(man))
    by_pid = {pid: g for pid, g in man.groupby("patient_id", sort=False)}

    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str)
    cases = cases.set_index("patient_id")

    split = pd.read_csv(split_csv)
    train_pids = list(split[split.split == "train"].patient_id.astype(str))
    val_pids = list(split[split.split == "val"].patient_id.astype(str))

    items = {}
    for pid in train_pids:
        row = cases.loc[pid]
        items[pid] = {
            "b3": B3[str(row.birads3)], "dens": DENS[str(row.density)],
            "q": {1: parse_zones(row.quadrant_right), 0: parse_zones(row.quadrant_left)},
            "idxs": [int(i) for i in by_pid[pid].index],
            "vi": [VIEW_ID[v] for v in by_pid[pid].view],
            "si": [side_id(l) for l in by_pid[pid].laterality],
        }

    log("loading backbone")
    bb = timm.create_model("convnext_small", pretrained=False)
    bb.head.fc = nn.Linear(768, 768)
    bb.birads_layer = nn.Sequential(nn.Linear(768, 5))
    bb.density_layer = nn.Sequential(nn.Linear(768, 4))
    bb.load_state_dict(torch.load(ROOT / "weights/convnext_small_ftune.pth", map_location="cpu", weights_only=False))
    bb = bb.to(device)
    for p in bb.parameters():
        p.requires_grad = False
    for p in bb.stages[-1].parameters():
        p.requires_grad = True
    head = HieraMIL3(768).to(device)
    stats = np.load(ROOT / "results/feat_stats.npz")

    opt = torch.optim.AdamW([
        {"params": [p for p in bb.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    rng = np.random.default_rng(0)
    metrics = {"mode": "partial-ft last stage", "train_cases": len(train_pids), "epochs": args.epochs}
    for ep in range(args.epochs):
        bb.train(); head.train()
        order = train_pids.copy(); rng.shuffle(order)
        t0 = time.time(); tot = 0.0; done = 0
        for start in range(0, len(order), args.batch_cases):
            pids = order[start:start + args.batch_cases]
            idxs = [i for pid in pids for i in items[pid]["idxs"]]
            x = batch_images(arr, idxs, device)
            vi = torch.tensor([v for pid in pids for v in items[pid]["vi"]], device=device)
            si = torch.tensor([s for pid in pids for s in items[pid]["si"]], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                f = bb.forward_head(bb.forward_features(x), pre_logits=True)
            f = (f.float() - torch.tensor(stats["mu"], device=device)) / torch.tensor(stats["sd"], device=device)
            off = 0
            loss = torch.tensor(0.0, device=device)
            for pid in pids:
                n = len(items[pid]["idxs"])
                lb, ld, lq = head(f[off:off + n], vi[off:off + n], si[off:off + n])
                loss = loss + gated_loss(lb, ld, lq, items[pid], device)
                off += n
            loss = loss / len(pids)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); done += len(pids)
            if done % 10 == 0 or done >= len(order):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                log(f"epoch {ep+1}/{args.epochs} {done}/{len(order)} | loss {tot/done:.4f} | {rate:.2f} case/s | ETA {(len(order)-done)/max(rate,1e-9)/60:.1f} min")
            if args.max_steps and done >= args.max_steps:
                break
        log(f"epoch {ep+1} done | avg loss {tot/max(done,1):.4f}")
        metrics[f"val_epoch{ep+1}"] = evaluate(val_pids, np.load(ROOT / "results/val_u8.npy", mmap_mode="r"),
                                               {pid: g for pid, g in pd.read_csv(ROOT / "results/val_manifest.csv").assign(
                                                   patient_id=lambda d: d.patient_id.astype(str)).groupby("patient_id", sort=False)},
                                               cases, bb, head, stats, device, "val")
        log("val epoch " + json.dumps(metrics[f"val_epoch{ep+1}"]))

    if args.skip_test:
        out = Path(args.out); out.mkdir(exist_ok=True)
        torch.save({"backbone": bb.state_dict(), "head": head.state_dict()}, out / "finetune_rocm_smoke.pt")
        (out / "finetune_rocm_smoke_metrics.json").write_text(json.dumps(metrics, indent=2))
        log("skip-test mode done")
        return

    log("evaluating YARISMA")
    yar = ROOT / "yarisma"
    man_y = pd.read_csv(ROOT / "labels/yarisma_manifest.csv")
    man_y["patient_id"] = man_y.patient_id.astype(str)
    by_y = {pid: g for pid, g in man_y.groupby("patient_id", sort=False)}
    test_pids = list(by_y)
    bb.eval(); head.eval()
    yb, pb, yd, pdd, hits, sides, scores = [], [], [], [], 0, 0, []
    t0 = time.time()
    for i, pid in enumerate(test_pids, 1):
        g = by_y[pid]
        xs = []
        for fn in g.image_filename:
            a = np.array(Image.open(yar / pid / fn).convert("L"), dtype=np.float32)
            a = a - a.min()
            a = a / max(float(a.max()), 1e-6)
            xs.append(np.stack([a, a, a]))
        x = torch.from_numpy(np.stack(xs)).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            f = bb.forward_head(bb.forward_features(x), pre_logits=True)
        f = ((f.float().cpu().numpy() - stats["mu"]) / stats["sd"])
        vi = torch.tensor([VIEW_ID[v] for v in g.view], device=device)
        si = torch.tensor([side_id(l) for l in g.laterality], device=device)
        with torch.no_grad():
            lb, ld, lq = head(torch.tensor(f, dtype=torch.float32, device=device), vi, si)
        row = cases.loc[pid]
        b, d = int(lb.argmax()), int(ld.argmax())
        yb.append(B3[str(row.birads3)]); pb.append(b); yd.append(DENS[str(row.density)]); pdd.append(d)
        qpred = {s: int(v.argmax()) for s, v in lq.items()}
        for s, key in ((1, "quadrant_right"), (0, "quadrant_left")):
            zs = parse_zones(row[key])
            if zs and s in qpred:
                sides += 1; hits += int(ZONES[qpred[s]] in zs)
        if b == B3[str(row.birads3)]:
            sc = 0.70 + 0.15 * float(d == DENS[str(row.density)])
            hh = [int(ZONES[qpred[s]] in parse_zones(row[k])) for s, k in ((1, "quadrant_right"), (0, "quadrant_left")) if parse_zones(row[k]) and s in qpred]
            sc += 0.15 * (float(np.mean(hh)) if hh else 1.0)
            scores.append(sc)
        else:
            scores.append(0.0)
        if i % 200 == 0:
            el = time.time() - t0; log(f"test {i}/{len(test_pids)} | {i/el:.1f} case/s")
    yb, pb, yd, pdd = np.array(yb), np.array(pb), np.array(yd), np.array(pdd)
    def f1(y, p, k):
        out = []
        for c in range(k):
            tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
            out.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(out)), [round(x, 3) for x in out]
    bf1, bper = f1(yb, pb, 3); df1, dper = f1(yd, pdd, 4)
    scores = np.array(scores)
    rng2 = np.random.default_rng(0)
    boots = np.array([scores[rng2.integers(0, len(scores), len(scores))].mean() for _ in range(1000)])
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    metrics["test"] = {"split": "yarisma", "n": len(test_pids), "birads_macro_f1": round(bf1, 4),
                       "birads_acc": round(float((yb == pb).mean()), 4), "birads_per_class_f1": bper,
                       "density_macro_f1": round(df1, 4), "density_acc": round(float((yd == pdd).mean()), 4),
                       "density_per_class_f1": dper, "quadrant_side_acc": round(hits / max(sides, 1), 4),
                       "quadrant_sides": sides, "gated_score": round(float(scores.mean()), 4),
                       "gated_ci95": [round(ci[0], 4), round(ci[1], 4)]}
    out = Path(args.out); out.mkdir(exist_ok=True)
    torch.save({"backbone": bb.state_dict(), "head": head.state_dict()}, out / "finetune_rocm.pt")
    (out / "finetune_rocm_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
