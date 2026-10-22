"""Combine the 3-seed and 5-seed soups; select the decision rule on val, report test once."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("MAMMO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from improve_batched import B3, DENS, FEAT, HieraMILB, infer, parse_zones, soup  # noqa: E402
from final_push import apply_tau, fast_gated, tune  # noqa: E402
from train_heads import load_view_meta  # noqa: E402
from ensemble_calibrate import evaluate  # noqa: E402


def log(*a):
    print(*a, flush=True)


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = HieraMILB(ck["dim"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    return m


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

    va = build([k for k in z.keys() if k.startswith("va_")])
    te = build([k for k in z.keys() if k.startswith("te_")])

    paths3 = [ROOT / f"results/improve/soup_ce_seed{i}.pt" for i in (0, 1, 2)]
    paths5 = [ROOT / f"results/improve/push_seed{i}.pt" for i in (0, 1, 2, 3, 4)]
    runs = {}
    for tag, paths in (("3", paths3), ("5", paths5), ("8", paths3 + paths5)):
        rv, rt = [], []
        for p in paths:
            m = load_model(p, device)
            rv.append(infer(m, va, device)); rt.append(infer(m, te, device))
        runs[tag] = (soup(rv), soup(rt))
        log(f"soup {tag}: {len(paths)} models inferred")

    out = {}
    best_tag, best_val = None, -1
    for tag, (sv, st) in runs.items():
        bv, bd, tau, val = tune(va, sv)
        test = evaluate(te, apply_tau(st, tau, bv), device, bias_b=bv, bias_d=bd, name=f"soup{tag}_tuned")
        out[tag] = {"val_tuned": val, "test": test, "bias_b": bv.tolist(), "bias_d": bd.tolist(), "tau": tau}
        log(f"soup {tag}: val {val:.4f} | test gated {test['gated_score']:.4f} | BI-RADS {test['birads_macro_f1']:.4f} | density {test['density_macro_f1']:.4f}")
        if val > best_val:
            best_val, best_tag = val, tag
    log(f"selected by val: soup {best_tag} (val {best_val:.4f}) -> test gated {out[best_tag]['test']['gated_score']:.4f}")
    p = ROOT / "results/push/combined_soup_metrics.json"
    p.write_text(json.dumps({"selected": best_tag, "results": out}, indent=2))
    log(f"saved {p}")


if __name__ == "__main__":
    main()
