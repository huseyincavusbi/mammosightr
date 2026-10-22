"""4-fold cross-validation for error bars (one model per fold, CE density)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("MAMMO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from improve_batched import B3, DENS, FEAT, HieraMILB, infer, parse_zones, train  # noqa: E402
from train_heads import load_view_meta  # noqa: E402
from ensemble_calibrate import evaluate  # noqa: E402


def log(*a):
    print(*a, flush=True)


def summarize(vals):
    a = np.array(vals, dtype=float)
    n = len(a)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)), 4),
            "values": [round(float(x), 4) for x in a],
            "ci95_t": [round(float(a.mean() - 3.182 * a.std(ddof=1) / np.sqrt(n)), 4),
                       round(float(a.mean() + 3.182 * a.std(ddof=1) / np.sqrt(n)), 4)]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    z = np.load(FEAT)
    cases = pd.read_csv(ROOT / "labels/cases.csv")
    cases["patient_id"] = cases.patient_id.astype(str); cases = cases.set_index("patient_id")
    folds = pd.read_csv(ROOT / "labels/folds.csv")
    folds["patient_id"] = folds.patient_id.astype(str)
    fold_of = dict(zip(folds.patient_id, folds.fold))
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

    pool = build([k for k in z.keys() if k.startswith(("tr_", "va_"))])
    te = build([k for k in z.keys() if k.startswith("te_")])
    log(f"pool {len(pool)} | test {len(te)}")
    missing = [it["pid"] for it in pool if it["pid"] not in fold_of]
    assert not missing, f"missing folds for {len(missing)} cases"

    out = ROOT / "results/cv4"; out.mkdir(parents=True, exist_ok=True)
    fold_metrics = []
    for f in (1, 2, 3, 4):
        tr = [it for it in pool if fold_of[it["pid"]] != f]
        va = [it for it in pool if fold_of[it["pid"]] == f]
        log(f"=== fold {f}: train {len(tr)} val {len(va)} ===")
        m = train(tr, seed=f, loss_type="ce", device=device, epochs=5, bs=256)
        torch.save({"state": m.state_dict(), "dim": 768}, out / f"cv4_fold{f}.pt")
        vp = infer(m, va, device); tp = infer(m, te, device)
        vm = evaluate(va, vp, device, name=f"fold{f}_val")
        tm = evaluate(te, tp, device, name=f"fold{f}_test")
        log(f"  fold {f} val:  " + json.dumps(vm))
        log(f"  fold {f} test: " + json.dumps(tm))
        fold_metrics.append({"fold": f, "val": vm, "test": tm})

    def agg(split, key):
        return summarize([fm[split][key] for fm in fold_metrics])

    summary = {}
    for split in ("val", "test"):
        summary[split] = {
            "birads_macro_f1": agg(split, "birads_macro_f1"),
            "density_macro_f1": agg(split, "density_macro_f1"),
            "quadrant_side_acc": agg(split, "quadrant_side_acc"),
            "gated_score": agg(split, "gated_score"),
        }
    (out / "cv4_metrics.json").write_text(json.dumps({"folds": fold_metrics, "summary": summary}, indent=2))
    log("=== SUMMARY ===")
    for split in ("val", "test"):
        for k, v in summary[split].items():
            log(f"{split:4s} {k:20s} mean {v['mean']:.4f} ± {v['std']:.4f}  (95% t-CI {v['ci95_t']})")
    log("saved results/cv4/cv4_metrics.json")


if __name__ == "__main__":
    main()
