from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.covariance import LedoitWolf


def window_to_20d(windows: np.ndarray) -> np.ndarray:
    mean = np.mean(windows, axis=1)
    std = np.std(windows, axis=1)
    return np.concatenate([mean, std], axis=1).astype(np.float64)


class HotellingT2:
    def __init__(
        self,
        mu0: Optional[np.ndarray] = None,
        precision: Optional[np.ndarray] = None,
    ) -> None:
        self.mu0 = mu0
        self.precision = precision

    def fit(self, windows: np.ndarray) -> "HotellingT2":
        v = window_to_20d(windows)
        lw = LedoitWolf().fit(v)
        self.mu0 = lw.location_.astype(np.float64)
        self.precision = lw.precision_.astype(np.float64)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.mu0 is None or self.precision is None:
            raise RuntimeError("HotellingT2 must be fitted before scoring.")
        if windows.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        v = window_to_20d(windows)
        diff = v - self.mu0
        t2 = np.sum((diff @ self.precision) * diff, axis=1)
        return np.maximum(t2, 0.0).astype(np.float32)

    def decompose(self, windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mu0 is None or self.precision is None:
            raise RuntimeError("HotellingT2 must be fitted before decompose.")
        if windows.shape[0] == 0:
            d = windows.shape[2] if windows.ndim == 3 else 10
            return np.zeros((0, d), dtype=np.float32), np.zeros((0, d), dtype=np.float32)
        v = window_to_20d(windows)
        diff = v - self.mu0
        contrib = diff * (diff @ self.precision)
        d = contrib.shape[1] // 2
        return contrib[:, :d].astype(np.float32), contrib[:, d:].astype(np.float32)

    def state_dict(self) -> Dict:
        return {
            "mu0": self.mu0,
            "precision": self.precision,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "HotellingT2":
        return cls(
            mu0=np.asarray(state["mu0"], dtype=np.float64),
            precision=np.asarray(state["precision"], dtype=np.float64),
        )
