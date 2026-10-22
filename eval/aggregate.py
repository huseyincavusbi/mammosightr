"""Case-level aggregation of per-image probabilities."""
import numpy as np


def aggregate_probs(probs, case_ids, method="mean", weights=None):
    probs = np.asarray(probs, dtype=float)
    case_ids = np.asarray(case_ids)
    order = list(dict.fromkeys(case_ids.tolist()))
    idxs = {c: [] for c in order}
    for i, c in enumerate(case_ids):
        idxs[c].append(i)
    out = np.zeros((len(order), probs.shape[1]))
    for j, c in enumerate(order):
        ii = idxs[c]
        p = probs[ii]
        if method == "mean":
            out[j] = p.mean(axis=0)
        elif method == "max":
            out[j] = p.max(axis=0)
        elif method == "attention":
            w = np.ones(len(ii)) if weights is None else np.asarray(weights, dtype=float)[ii]
            s = w.sum()
            w = w / s if s > 0 else np.full(len(ii), 1.0 / len(ii))
            out[j] = (p * w[:, None]).sum(axis=0)
        else:
            raise ValueError(method)
    return order, out
