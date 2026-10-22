"""End-to-end inference: 4 mammogram PNGs -> BI-RADS + density + quadrant."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_heads import HieraMIL3, SIDE_ID, VIEW_ID, ZONES, DENS  # noqa: E402

B3_NAMES = ["0", "1-2", "4-5"]
DENS_NAMES = ["A", "B", "C", "D"]
FILES = {"LCC": (0, 0), "LMLO": (1, 0), "RCC": (0, 1), "RMLO": (1, 1)}


def load_backbone(device):
    m = timm.create_model("convnext_small", pretrained=False)
    m.head.fc = nn.Linear(768, 768)
    m.birads_layer = nn.Sequential(nn.Linear(768, 5))
    m.density_layer = nn.Sequential(nn.Linear(768, 4))
    m.load_state_dict(torch.load(ROOT / "code/MAM-CLIP/weights/convnext_small_ftune.pth",
                                 map_location="cpu", weights_only=False))
    return m.eval().to(device)


def preprocess(path):
    a = np.array(Image.open(path).convert("L"))
    if a.shape != (1024, 768):
        import cv2
        a = cv2.resize(a, (768, 1024))
    x = a.astype(np.float32)
    x = x - x.min()
    x = x / max(float(x.max()), 1e-6)
    return np.stack([x, x, x])


@torch.no_grad()
def forward_case(backbone, head, stats, paths, device):
    xs = np.stack([preprocess(p) for p in paths])
    feats = backbone.forward_head(backbone.forward_features(torch.from_numpy(xs).to(device)),
                                 pre_logits=True).float().cpu().numpy()
    feats = (feats - stats["mu"]) / stats["sd"]
    n = len(paths)
    vi = torch.tensor([v for v, _ in [FILES[Path(p).stem.upper()] for p in paths]], device=device)
    si = torch.tensor([s for _, s in [FILES[Path(p).stem.upper()] for p in paths]], device=device)
    lb, ld, lq = head(torch.tensor(feats, dtype=torch.float32, device=device), vi, si)
    assert n == len(vi)
    return lb, ld, lq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="case folder with LCC/LMLO/RCC/RMLO pngs")
    ap.add_argument("--case", help="patient id (uses cached YARISMA features)")
    ap.add_argument("--head", default=str(ROOT / "assets/heads/soup_ce_seed0.pt"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    head = HieraMIL3(ck["dim"]).to(device)
    head.load_state_dict(ck["state"])
    head.eval()
    stats_path = ROOT / "assets/feat_stats.npz"
    if not stats_path.exists():
        stats_path = ROOT / "results/feat_stats.npz"
    stats = np.load(stats_path)

    if args.case:
        z = np.load(ROOT / "notebooks/out/features_full_gulluk_convnext_small.npz")
        key = f"te_{args.case}"
        feats = torch.tensor(z[key], dtype=torch.float32, device=device)
        import pandas as pd
        man = pd.read_csv(ROOT / "labels/yarisma_manifest.csv")
        man["patient_id"] = man.patient_id.astype(str)
        rows = man[man.patient_id == args.case]
        vi = torch.tensor([VIEW_ID[r.view] for r in rows.itertuples()], device=device)
        si = torch.tensor([SIDE_ID[r.laterality] for r in rows.itertuples()], device=device)
        with torch.no_grad():
            lb, ld, lq = head(feats, vi, si)
    else:
        files = {}
        for p in Path(args.dir).iterdir():
            stem = p.stem.upper()
            if stem in FILES:
                files[stem] = p
        missing = set(FILES) - set(files)
        if missing:
            raise SystemExit(f"missing views: {sorted(missing)}")
        paths = [files[k] for k in ("LCC", "LMLO", "RCC", "RMLO")]
        backbone = load_backbone(device)
        lb, ld, lq = forward_case(backbone, head, stats, paths, device)

    pb = torch.softmax(lb, 0).cpu().numpy()
    pdens = torch.softmax(ld, 0).cpu().numpy()
    out = {
        "birads": B3_NAMES[int(pb.argmax())],
        "birads_probs": {k: round(float(v), 4) for k, v in zip(B3_NAMES, pb)},
        "density": DENS_NAMES[int(pdens.argmax())],
        "density_probs": {k: round(float(v), 4) for k, v in zip(DENS_NAMES, pdens)},
    }
    for s, name in ((1, "right"), (0, "left")):
        if s in lq:
            pq = torch.sigmoid(lq[s]).cpu().numpy()
            out[f"quadrant_{name}"] = ZONES[int(pq.argmax())]
            out[f"quadrant_{name}_probs"] = {k: round(float(v), 4) for k, v in zip(ZONES, pq)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
