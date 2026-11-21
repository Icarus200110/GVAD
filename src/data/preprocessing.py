from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURE_NAMES: List[str] = [
    "QUAL",
    "MQ",
    "DP",
    "QD",
    "SOR",
    "AF",
    "HIAF",
    "PMEAN",
    "PSTD",
    "LOG_DPOS",
]


def compute_sample_qc_metrics(records: List[Dict]) -> np.ndarray:
    qual = np.array([r["QUAL"] for r in records], dtype=np.float64)
    mq = np.array([r["MQ"] for r in records], dtype=np.float64)
    dp = np.array([r["DP"] for r in records], dtype=np.float64)
    qd = np.array([r["QD"] for r in records], dtype=np.float64)
    sor = np.array([r["SOR"] for r in records], dtype=np.float64)
    af = np.array([r["AF"] for r in records], dtype=np.float64)
    flt = [r.get("filter", "PASS") for r in records]

    finite_sor = sor[np.isfinite(sor)]
    sor_cap = float(np.percentile(finite_sor, 99)) if finite_sor.size > 0 else 10.0
    sor_clean = np.where(np.isfinite(sor), sor, sor_cap)

    dp_mean = float(np.mean(dp)) if dp.size > 0 else 1.0
    dp_cv = float(np.std(dp) / max(dp_mean, 1e-8))
    low_af_frac = float(np.mean(af < 0.05))
    high_sor_frac = float(np.mean(sor_clean > 3.0))
    pass_frac = float(np.mean([1.0 if f in {"PASS", "."} else 0.0 for f in flt]))

    return np.array(
        [
            float(np.median(qual)),
            float(np.median(mq)),
            float(np.median(qd)),
            float(np.median(dp)),
            dp_cv,
            low_af_frac,
            high_sor_frac,
            pass_frac,
        ],
        dtype=np.float64,
    )


def screen_baseline_outliers(
    samples: List[Dict],
    z_thresh: float = 3.5,
    min_flagged: int = 2,
) -> Tuple[List[Dict], List[str]]:
    if len(samples) < 3:
        return list(samples), []

    metric_matrix = np.vstack([compute_sample_qc_metrics(s["records"]) for s in samples])
    med = np.median(metric_matrix, axis=0)
    mad = np.median(np.abs(metric_matrix - med), axis=0)
    mad_safe = np.where(mad < 1e-8, 1e-8, mad)
    modified_z = 0.6745 * (metric_matrix - med) / mad_safe
    flagged_counts = np.sum(np.abs(modified_z) > z_thresh, axis=1)

    clean_samples: List[Dict] = []
    removed_ids: List[str] = []
    for sample, count in zip(samples, flagged_counts):
        if int(count) >= min_flagged:
            removed_ids.append(sample["sample_id"])
        else:
            clean_samples.append(sample)
    return clean_samples, removed_ids


def extract_raw_features(records: List[Dict]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    n = len(records)
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    chroms: List[str] = []
    positions = np.zeros(n, dtype=np.int64)

    prev_chrom: Optional[str] = None
    prev_pos: int = 0

    for i, r in enumerate(records):
        chrom = r["chrom"]
        pos = int(r["pos"])
        chroms.append(chrom)
        positions[i] = pos

        if prev_chrom == chrom:
            dpos = max(0, pos - prev_pos)
        else:
            dpos = 0
        prev_chrom = chrom
        prev_pos = pos

        qual = float(r["QUAL"])
        dp = float(r["DP"])
        qd = float(r.get("QD", qual / max(dp, 1.0)))

        features[i, 0] = qual
        features[i, 1] = float(r["MQ"])
        features[i, 2] = dp
        features[i, 3] = qd
        features[i, 4] = float(r["SOR"])
        features[i, 5] = float(r["AF"])
        features[i, 6] = float(r["HIAF"])
        features[i, 7] = float(r["PMEAN"])
        features[i, 8] = float(r["PSTD"])
        features[i, 9] = float(np.log1p(dpos))

    return features, chroms, positions


class FeatureScaler:
    def __init__(
        self,
        sor_clip: float = 10.0,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.sor_clip = float(sor_clip)
        self.mean = mean
        self.std = std

    def fit(self, raw_feature_list: List[np.ndarray]) -> "FeatureScaler":
        stacked = np.vstack(raw_feature_list).astype(np.float64)
        sor_col = stacked[:, 4]
        finite_sor = sor_col[np.isfinite(sor_col)]
        self.sor_clip = float(np.percentile(finite_sor, 99)) if finite_sor.size > 0 else 10.0

        cleaned = self._clean_matrix(stacked)
        self.mean = np.mean(cleaned, axis=0).astype(np.float32)
        std = np.std(cleaned, axis=0)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return self

    def _clean_matrix(self, raw: np.ndarray) -> np.ndarray:
        out = raw.astype(np.float64, copy=True)
        sor = out[:, 4]
        sor = np.where(np.isposinf(sor), self.sor_clip, sor)
        sor = np.where(np.isneginf(sor), 0.0, sor)
        sor = np.clip(sor, 0.0, self.sor_clip)
        out[:, 4] = sor
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler must be fitted before transform.")
        cleaned = self._clean_matrix(raw).astype(np.float32)
        return (cleaned - self.mean) / self.std

    def state_dict(self) -> Dict:
        return {
            "sor_clip": self.sor_clip,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "FeatureScaler":
        return cls(
            sor_clip=float(state["sor_clip"]),
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


def build_windows(
    features: np.ndarray,
    chroms: List[str],
    positions: np.ndarray,
    window_size: int = 16,
    stride: int = 8,
) -> Tuple[np.ndarray, List[Dict]]:
    windows: List[np.ndarray] = []
    meta: List[Dict] = []
    n = features.shape[0]
    if n < window_size:
        return np.zeros((0, window_size, features.shape[1]), dtype=np.float32), meta

    start = 0
    while start < n:
        chrom = chroms[start]
        end = start
        while end < n and chroms[end] == chrom:
            end += 1

        idx = start
        while idx + window_size <= end:
            w = features[idx : idx + window_size]
            windows.append(w)
            meta.append(
                {
                    "chrom": chrom,
                    "start_pos": int(positions[idx]),
                    "end_pos": int(positions[idx + window_size - 1]),
                    "start_idx": int(idx),
                    "end_idx": int(idx + window_size),
                }
            )
            idx += stride

        start = end

    if not windows:
        return np.zeros((0, window_size, features.shape[1]), dtype=np.float32), meta
    return np.stack(windows, axis=0).astype(np.float32), meta
