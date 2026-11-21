from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def _threshold_for_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr + 1e-12)[0]
    idx = int(valid[-1]) if valid.size > 0 else 0
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def detection_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fprs: Sequence[float] = (0.02, 0.05, 0.10, 0.15, 0.20),
) -> List[Dict]:
    rows = []
    for tfpr in target_fprs:
        thresh, actual_fpr, det_rate = _threshold_for_fpr(y_true, y_score, float(tfpr))
        rows.append(
            {
                "target_fpr": float(tfpr),
                "actual_fpr": actual_fpr,
                "detection_rate": det_rate,
                "threshold": thresh,
            }
        )
    return rows


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    fpr = float(fp / max(fp + tn, 1))
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def stratify_by_severity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    severities: Sequence[Optional[float]],
    target_fpr: float = 0.10,
    num_bins: int = 5,
) -> List[Dict]:
    thresh, _, _ = _threshold_for_fpr(y_true, y_score, target_fpr)
    anom_indices = [i for i, y in enumerate(y_true) if y == 1 and severities[i] is not None]
    if not anom_indices:
        return []

    anom_sev = np.array([float(severities[i]) for i in anom_indices])
    anom_scores = np.array([float(y_score[i]) for i in anom_indices])
    quantiles = np.quantile(anom_sev, np.linspace(0.0, 1.0, num_bins + 1))

    rows = []
    for b in range(num_bins):
        low = float(quantiles[b])
        high = float(quantiles[b + 1])
        if b < num_bins - 1:
            mask = (anom_sev >= low) & (anom_sev < high)
        else:
            mask = (anom_sev >= low) & (anom_sev <= high)
        count = int(np.sum(mask))
        detected = int(np.sum(anom_scores[mask] >= thresh)) if count > 0 else 0
        rows.append(
            {
                "severity_min": low,
                "severity_max": high,
                "count": count,
                "detected": detected,
                "detection_rate": float(detected / max(count, 1)),
            }
        )
    return rows


def stratify_by_profile(
    y_true: np.ndarray,
    y_score: np.ndarray,
    profiles: Sequence[Optional[str]],
    target_fpr: float = 0.10,
) -> Dict[str, Dict]:
    thresh, _, _ = _threshold_for_fpr(y_true, y_score, target_fpr)
    summary: Dict[str, Dict] = {}
    for prof in ("step", "ramp"):
        idx = [i for i, p in enumerate(profiles) if y_true[i] == 1 and p == prof]
        count = len(idx)
        detected = int(sum(1 for i in idx if y_score[i] >= thresh))
        summary[prof] = {
            "count": count,
            "detected": detected,
            "detection_rate": float(detected / max(count, 1)),
        }
    return summary
