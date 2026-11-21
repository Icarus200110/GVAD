import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import screen_baseline_outliers
from src.data.synthesis import apply_degradation_to_records
from src.data.vcf_reader import load_directory
from src.evaluation.metrics import (
    compute_binary_metrics,
    confusion_at_threshold,
    detection_at_fpr,
    stratify_by_profile,
    stratify_by_severity,
)
from src.models.ewma import ewma_trajectory
from src.models.gvad import GVADDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gvad.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_directory(args.input_dir, sample_index=args.sample_index)
    samples, _ = screen_baseline_outliers(samples)
    if len(samples) < 4:
        raise SystemExit("Need at least 4 samples to split test-normal and synth-anomaly sets.")

    rng = np.random.default_rng(args.seed)
    mid = len(samples) // 2
    test_normal = samples[:mid]
    synth_base = samples[mid:]

    detector = GVADDetector.load(args.checkpoint, device=args.device)

    y_true = []
    gvad_scores = []
    ae_only_scores = []
    severities = []
    profiles = []
    normal_trajs = []
    anom_trajs = []

    for s in test_normal:
        res = detector.score_sample(s["records"], sample_id=s["sample_id"], device=args.device)
        _, ae_qmax = ewma_trajectory(res["ae_scores"], alpha=detector.alpha)
        y_true.append(0)
        gvad_scores.append(res["q_max"])
        ae_only_scores.append(ae_qmax)
        severities.append(None)
        profiles.append(None)
        if len(normal_trajs) < 5:
            normal_trajs.append(
                {
                    "sample_id": s["sample_id"],
                    "q_traj": res["q_traj"].tolist(),
                }
            )

    for s in synth_base:
        perturbed_records, info = apply_degradation_to_records(s["records"], rng=rng)
        res = detector.score_sample(
            perturbed_records,
            sample_id=f"{s['sample_id']}_anom",
            device=args.device,
        )
        _, ae_qmax = ewma_trajectory(res["ae_scores"], alpha=detector.alpha)
        y_true.append(1)
        gvad_scores.append(res["q_max"])
        ae_only_scores.append(ae_qmax)
        severities.append(info["severity"])
        profiles.append(info["profile"])
        if len(anom_trajs) < 5:
            onset_win = int(info["onset_frac"] * max(1, res["num_windows"]))
            anom_trajs.append(
                {
                    "sample_id": f"{s['sample_id']}_anom",
                    "profile": info["profile"],
                    "onset_window": onset_win,
                    "q_traj": res["q_traj"].tolist(),
                }
            )

    y_arr = np.asarray(y_true, dtype=int)
    gvad_arr = np.asarray(gvad_scores, dtype=np.float64)
    ae_arr = np.asarray(ae_only_scores, dtype=np.float64)

    results = {
        "warn_limit": detector.ewma.warn_limit,
        "crit_limit": detector.ewma.crit_limit,
        "gvad_roc": compute_binary_metrics(y_arr, gvad_arr),
        "ae_roc": compute_binary_metrics(y_arr, ae_arr),
        "fpr_operating_points": detection_at_fpr(y_arr, gvad_arr),
        "confusion_at_crit": confusion_at_threshold(y_arr, gvad_arr, detector.ewma.crit_limit),
        "severity_stratification": stratify_by_severity(y_arr, gvad_arr, severities),
        "profile_stratification": stratify_by_profile(y_arr, gvad_arr, profiles),
        "example_normal_trajs": normal_trajs,
        "example_anomaly_trajs": anom_trajs,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"GVAD ROC-AUC: {results['gvad_roc']['roc_auc']:.4f} | "
        f"AE-only ROC-AUC: {results['ae_roc']['roc_auc']:.4f}"
    )
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
