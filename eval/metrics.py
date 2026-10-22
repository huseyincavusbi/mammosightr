"""Image- and case-level metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def macro_f1(y_true, y_pred, labels=None):
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, labels=list(labels),
                                 output_dict=True, zero_division=0)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def ece(probs, y_true, n_bins=15):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    val = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        val += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(val)
