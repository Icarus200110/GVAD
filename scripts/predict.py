import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.vcf_reader import load_directory, load_variant_file
from src.models.gvad import GVADDetector, annotate_vcf_filter, export_flagged_bed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gvad.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--annotate-vcf", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = GVADDetector.load(args.checkpoint, device=args.device)

    if input_path.is_dir():
        samples = load_directory(input_path, sample_index=args.sample_index)
    else:
        records = load_variant_file(input_path, sample_index=args.sample_index)
        samples = [
            {
                "sample_id": input_path.stem.replace(".vcf", ""),
                "path": str(input_path),
                "records": records,
            }
        ]

    summaries = []
    for s in samples:
        res = detector.score_sample(s["records"], sample_id=s["sample_id"], device=args.device)
        export_flagged_bed(res, detector.ewma.warn_limit, out_dir / f"{s['sample_id']}.flagged.bed")

        if args.annotate_vcf and str(s["path"]).lower().endswith((".vcf", ".vcf.gz")):
            annotate_vcf_filter(
                input_vcf=s["path"],
                result=res,
                warn_limit=detector.ewma.warn_limit,
                crit_limit=detector.ewma.crit_limit,
                output_vcf=out_dir / f"{s['sample_id']}.gvad.vcf",
            )

        summary = {
            "sample_id": res["sample_id"],
            "num_variants": res["num_variants"],
            "num_windows": res["num_windows"],
            "q_max": res["q_max"],
            "alert": res["alert"],
            "breach_idx": res["breach_idx"],
            "breach_region": res["breach_region"],
            "feature_contributions": res["feature_contributions"],
        }
        summaries.append(summary)
        print(f"{res['sample_id']}: alert={res['alert']} Q_max={res['q_max']:.4f}")

    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
