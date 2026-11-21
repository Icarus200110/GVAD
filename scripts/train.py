import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import screen_baseline_outliers
from src.data.vcf_reader import load_directory
from src.models.gvad import GVADDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/gvad.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--skip-mad", action="store_true")
    parser.add_argument("--ae-epochs", type=int, default=60)
    parser.add_argument("--gate-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_directory(args.input_dir, sample_index=args.sample_index)
    if not samples:
        raise SystemExit(f"No valid variant files found in {args.input_dir}")

    if not args.skip_mad:
        clean_samples, removed = screen_baseline_outliers(samples)
        print(f"Loaded {len(samples)} samples; MAD removed {len(removed)}; kept {len(clean_samples)}.")
        samples = clean_samples

    detector = GVADDetector(window_size=16, stride=8, alpha=0.2)
    detector.fit(
        samples=samples,
        ae_epochs=args.ae_epochs,
        gate_epochs=args.gate_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    detector.save(args.output)
    print(
        f"Saved GVAD checkpoint to {args.output} "
        f"(warn={detector.ewma.warn_limit:.4f}, crit={detector.ewma.crit_limit:.4f})"
    )


if __name__ == "__main__":
    main()
