"""Gradio demo: 4 mammogram views -> BI-RADS + density + quadrant."""
import json
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import B3_NAMES, DENS_NAMES, FILES, load_backbone, forward_case  # noqa: E402
from train_heads import HieraMIL3, ZONES  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
HEADS_DIR = ROOT / "assets/heads"
if not HEADS_DIR.exists():
    HEADS_DIR = ROOT / "results/improve"
HEADS = []
for _s in (0, 1, 2):
    ck = torch.load(HEADS_DIR / f"soup_ce_seed{_s}.pt", map_location="cpu", weights_only=False)
    _h = HieraMIL3(ck["dim"]).to(DEV)
    _h.load_state_dict(ck["state"])
    _h.eval()
    HEADS.append(_h)
BACKBONE = load_backbone(DEV)
STATS_PATH = ROOT / "assets/feat_stats.npz"
if not STATS_PATH.exists():
    STATS_PATH = ROOT / "results/feat_stats.npz"
STATS = np.load(STATS_PATH)


def predict(lcc, lmlo, rcc, rmlo):
    imgs = {"LCC": lcc, "LMLO": lmlo, "RCC": rcc, "RMLO": rmlo}
    if any(v is None for v in imgs.values()):
        return "—", "—", "—", "—", "Please provide all 4 views."
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for name in ("LCC", "LMLO", "RCC", "RMLO"):
            p = Path(td) / f"{name}.png"
            Image.fromarray(np.array(imgs[name])).convert("L").save(p)
            paths.append(str(p))
        lb_, ld_, lq_ = [], [], []
        for _h in HEADS:
            _lb, _ld, _lq = forward_case(BACKBONE, _h, STATS, paths, DEV)
            lb_.append(_lb); ld_.append(_ld); lq_.append(_lq)
        lb = torch.stack(lb_).mean(0)
        ld = torch.stack(ld_).mean(0)
        lq = {s: torch.stack([d[s] for d in lq_]).mean(0) for s in lq_[0]}
    pb = torch.softmax(lb, 0).cpu().numpy()
    pd_ = torch.softmax(ld, 0).cpu().numpy()
    birads = f"{B3_NAMES[int(pb.argmax())]}  ({pb.max():.0%})"
    dens = f"{DENS_NAMES[int(pd_.argmax())]}  ({pd_.max():.0%})"
    qr = ql = "—"
    details = {"birads_probs": {k: round(float(v), 3) for k, v in zip(B3_NAMES, pb)},
               "density_probs": {k: round(float(v), 3) for k, v in zip(DENS_NAMES, pd_)}}
    for s, name in ((1, "right"), (0, "left")):
        if s in lq:
            pq = torch.sigmoid(lq[s]).cpu().numpy()
            zone = f"{ZONES[int(pq.argmax())]}  ({pq.max():.0%})"
            details[f"quadrant_{name}_probs"] = {k: round(float(v), 3) for k, v in zip(ZONES, pq)}
            if name == "right":
                qr = zone
            else:
                ql = zone
    return birads, dens, ql, qr, json.dumps(details, indent=2)


EX_PIDS = ["845285109", "845285115", "836187020", "836194671", "825898394", "825898399"]
EXAMPLES = [[str(ROOT / f"yarisma/{p}/{n}.png") for n in ("LCC", "LMLO", "RCC", "RMLO")] for p in EX_PIDS]

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(image_mode="L", label="LCC"), gr.Image(image_mode="L", label="LMLO"),
            gr.Image(image_mode="L", label="RCC"), gr.Image(image_mode="L", label="RMLO")],
    outputs=[gr.Textbox(label="BI-RADS (0 / 1-2 / 4-5)"), gr.Textbox(label="Density (A-D)"),
             gr.Textbox(label="Quadrant left"), gr.Textbox(label="Quadrant right"), gr.JSON(label="Probabilities")],
    examples=EXAMPLES,
    cache_examples=False,
    title="MammosighTR — case-level BI-RADS / density / quadrant",
    description="Frozen Gulluk ConvNeXt-s + hierarchical attention MIL heads. Research use only, not a medical device.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
