from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np


DEGRADATION_MODES: List[str] = [
    "quality_collapse",
    "depth_shift",
    "strand_bias_surge",
    "fp_cluster",
    "af_drift",
    "caller_dispersion",
]


def apply_degradation_to_records(
    records: List[Dict],
    rng: np.random.Generator,
    onset_range: Tuple[float, float] = (0.30, 0.65),
    severity_range: Tuple[float, float] = (0.27, 0.99),
    profile: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    perturbed = deepcopy(records)
    n = len(perturbed)
    if n == 0:
        return perturbed, {}

    onset_frac = float(rng.uniform(*onset_range))
    onset_idx = min(max(1, int(n * onset_frac)), n - 1)
    severity = float(rng.uniform(*severity_range))
    if profile is None:
        profile = "step" if rng.random() < 0.4 else "ramp"

    num_modes = int(rng.integers(2, 5))
    modes = list(rng.choice(DEGRADATION_MODES, size=num_modes, replace=False))

    depth_up = bool(rng.random() < 0.5)
    depth_up_factor = float(rng.uniform(3.0, 6.0))
    depth_down_factor = float(rng.uniform(0.10, 0.30))
    qual_drop_base = float(rng.uniform(0.20, 0.80))
    sor_target = float(rng.uniform(5.0, 12.0))
    af_shift_base = float(rng.uniform(0.15, 0.35))
    pstd_factor = float(rng.uniform(2.0, 3.0))

    span = max(1, n - onset_idx - 1)
    for i in range(onset_idx, n):
        r = perturbed[i]
        prog = 1.0 if profile == "step" else float((i - onset_idx) / span)
        eff = severity * prog

        if "quality_collapse" in modes:
            scale = max(0.05, 1.0 - qual_drop_base * eff)
            r["QUAL"] *= scale
            r["MQ"] *= scale
            r["PMEAN"] *= scale

        if "depth_shift" in modes:
            if depth_up:
                r["DP"] *= 1.0 + (depth_up_factor - 1.0) * eff
            else:
                r["DP"] *= 1.0 - (1.0 - depth_down_factor) * eff

        if "strand_bias_surge" in modes:
            cur_sor = float(r["SOR"]) if np.isfinite(r["SOR"]) else sor_target
            r["SOR"] = cur_sor + (sor_target - cur_sor) * eff

        if "fp_cluster" in modes:
            fp_af = float(rng.uniform(0.01, 0.04))
            r["AF"] = float(r["AF"] * (1.0 - eff) + fp_af * eff)
            r["HIAF"] = float(r["HIAF"] * (1.0 - eff) + fp_af * eff)
            r["QUAL"] *= max(0.1, 1.0 - 0.6 * eff)

        if "af_drift" in modes:
            r["AF"] = float(np.clip(r["AF"] + af_shift_base * eff, 0.0, 1.0))
            r["HIAF"] = float(np.clip(r["HIAF"] + af_shift_base * eff, 0.0, 1.0))

        if "caller_dispersion" in modes:
            r["PSTD"] *= 1.0 + (pstd_factor - 1.0) * eff
            r["PSTD"] += float(rng.normal(0.0, 0.5 * eff))

        r["QD"] = float(r["QUAL"]) / max(float(r["DP"]), 1.0)

    info = {
        "onset_frac": onset_frac,
        "onset_idx": onset_idx,
        "severity": severity,
        "profile": profile,
        "modes": modes,
    }
    return perturbed, info


def perturb_windows(
    windows: np.ndarray,
    rng: np.random.Generator,
    frac: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    n_total = windows.shape[0]
    n_anom = max(1, int(n_total * frac))
    norm_idx = rng.choice(n_total, size=n_anom, replace=True)
    anom_idx = rng.choice(n_total, size=n_anom, replace=True)

    normal_w = windows[norm_idx].copy()
    anom_w = windows[anom_idx].copy()

    for i in range(n_anom):
        severity = float(rng.uniform(0.3, 1.0))
        num_modes = int(rng.integers(2, 5))
        modes = rng.choice(DEGRADATION_MODES, size=num_modes, replace=False)

        if "quality_collapse" in modes:
            drop = float(rng.uniform(0.8, 2.5)) * severity
            anom_w[i, :, 0] -= drop
            anom_w[i, :, 1] -= drop
            anom_w[i, :, 7] -= drop
            anom_w[i, :, 3] -= 0.7 * drop

        if "depth_shift" in modes:
            shift = float(rng.uniform(1.0, 3.0)) * severity
            if rng.random() < 0.5:
                anom_w[i, :, 2] += shift
            else:
                anom_w[i, :, 2] -= shift

        if "strand_bias_surge" in modes:
            anom_w[i, :, 4] += float(rng.uniform(1.2, 3.5)) * severity

        if "fp_cluster" in modes:
            anom_w[i, :, 5] -= float(rng.uniform(0.6, 1.8)) * severity
            anom_w[i, :, 0] -= float(rng.uniform(0.5, 1.5)) * severity

        if "af_drift" in modes:
            shift = float(rng.uniform(1.0, 2.8)) * severity
            anom_w[i, :, 5] += shift
            anom_w[i, :, 6] += shift

        if "caller_dispersion" in modes:
            anom_w[i, :, 8] += rng.normal(0.0, 1.5 * severity, size=anom_w.shape[1])

    return normal_w.astype(np.float32), anom_w.astype(np.float32)
