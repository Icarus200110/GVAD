from typing import Dict, List, Optional, Tuple

import numpy as np


def ewma_trajectory(scores: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32), 0.0
    q = np.zeros_like(scores, dtype=np.float32)
    q[0] = scores[0]
    one_minus = 1.0 - alpha
    for t in range(1, scores.shape[0]):
        q[t] = alpha * scores[t] + one_minus * q[t - 1]
    return q, float(np.max(q))


class EWMACalibrator:
    def __init__(
        self,
        alpha: float = 0.2,
        warn_limit: float = 0.0,
        crit_limit: float = 0.0,
    ) -> None:
        self.alpha = float(alpha)
        self.warn_limit = float(warn_limit)
        self.crit_limit = float(crit_limit)

    def fit(
        self,
        sample_scores_list: List[np.ndarray],
        warn_percentile: float = 90.0,
        crit_percentile: float = 99.0,
    ) -> "EWMACalibrator":
        q_max_values = []
        for s in sample_scores_list:
            if s.size > 0:
                _, q_max = ewma_trajectory(s, alpha=self.alpha)
                q_max_values.append(q_max)
        if not q_max_values:
            self.warn_limit = 0.0
            self.crit_limit = 0.0
            return self
        arr = np.asarray(q_max_values, dtype=np.float64)
        self.warn_limit = float(np.percentile(arr, warn_percentile))
        self.crit_limit = float(np.percentile(arr, crit_percentile))
        return self

    def evaluate_sample(self, scores: np.ndarray) -> Dict:
        q_traj, q_max = ewma_trajectory(scores, alpha=self.alpha)
        if q_max >= self.crit_limit:
            alert = "Red"
        elif q_max >= self.warn_limit:
            alert = "Yellow"
        else:
            alert = "Green"

        breach_idx: Optional[int] = None
        breach_candidates = np.where(q_traj >= self.warn_limit)[0]
        if breach_candidates.size > 0:
            breach_idx = int(breach_candidates[0])

        return {
            "q_traj": q_traj,
            "q_max": q_max,
            "alert": alert,
            "breach_idx": breach_idx,
        }

    def state_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "warn_limit": self.warn_limit,
            "crit_limit": self.crit_limit,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "EWMACalibrator":
        return cls(
            alpha=float(state["alpha"]),
            warn_limit=float(state["warn_limit"]),
            crit_limit=float(state["crit_limit"]),
        )
