"""Official competition score: 0.70*BI-RADS + 0.15*density + 0.15*quadrant,
with density/quadrant credit voided when BI-RADS is wrong.

quadrant_missing policy for cases without a true quadrant:
  "full" - quadrant credit is granted (case counts as correct)
  "skip" - score is renormalized over BI-RADS+density only
"""
import numpy as np

WEIGHTS = (0.70, 0.15, 0.15)


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() in ("", "nan", "None", "null")


def gated_scores(birads_pred, density_pred, quadrant_pred, birads_true, density_true,
                 quadrant_true, weights=WEIGHTS, quadrant_missing="full"):
    w0, w1, w2 = weights
    n = len(birads_true)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        if birads_pred[i] != birads_true[i]:
            continue
        s = w0
        if density_pred[i] == density_true[i]:
            s += w1
        if _is_missing(quadrant_true[i]):
            if quadrant_missing == "full":
                s += w2
            elif quadrant_missing == "skip":
                s = (w0 + (w1 if density_pred[i] == density_true[i] else 0.0)) / (w0 + w1)
            else:
                raise ValueError(quadrant_missing)
        elif quadrant_pred[i] == quadrant_true[i]:
            s += w2
        out[i] = s
    return out


def gated_score(*args, **kwargs):
    return float(np.mean(gated_scores(*args, **kwargs)))


def bootstrap_ci(birads_pred, density_pred, quadrant_pred, birads_true, density_true,
                 quadrant_true, weights=WEIGHTS, quadrant_missing="full",
                 n_boot=1000, seed=0, alpha=0.05):
    vals = gated_scores(birads_pred, density_pred, quadrant_pred, birads_true,
                        density_true, quadrant_true, weights, quadrant_missing)
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)
