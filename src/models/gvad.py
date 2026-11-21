import gzip
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.preprocessing import (
    FEATURE_NAMES,
    FeatureScaler,
    build_windows,
    extract_raw_features,
)
from src.models.autoencoder import Conv1DAutoencoder, score_autoencoder, train_autoencoder
from src.models.ewma import EWMACalibrator
from src.models.gating import GateFusion, train_gate
from src.models.hotelling import HotellingT2


class GVADDetector:
    def __init__(
        self,
        window_size: int = 16,
        stride: int = 8,
        alpha: float = 0.2,
        scaler: Optional[FeatureScaler] = None,
        t2_model: Optional[HotellingT2] = None,
        ae_model: Optional[Conv1DAutoencoder] = None,
        gate: Optional[GateFusion] = None,
        ewma: Optional[EWMACalibrator] = None,
    ) -> None:
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.alpha = float(alpha)
        self.scaler = scaler or FeatureScaler()
        self.t2_model = t2_model or HotellingT2()
        self.ae_model = ae_model
        self.gate = gate
        self.ewma = ewma or EWMACalibrator(alpha=alpha)

    def fit(
        self,
        samples: List[Dict],
        ae_epochs: int = 60,
        gate_epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        margin: float = 0.1,
        anom_frac: float = 0.20,
        seed: int = 42,
        device: str = "cpu",
    ) -> "GVADDetector":
        raw_list = []
        chroms_list = []
        pos_list = []
        for s in samples:
            raw, chroms, pos = extract_raw_features(s["records"])
            raw_list.append(raw)
            chroms_list.append(chroms)
            pos_list.append(pos)

        self.scaler.fit(raw_list)

        sample_windows = []
        for raw, chroms, pos in zip(raw_list, chroms_list, pos_list):
            norm_feat = self.scaler.transform(raw)
            w, _ = build_windows(
                norm_feat,
                chroms,
                pos,
                window_size=self.window_size,
                stride=self.stride,
            )
            sample_windows.append(w)

        non_empty = [w for w in sample_windows if w.shape[0] > 0]
        if not non_empty:
            raise ValueError("No valid windows constructed from training samples.")
        all_windows = np.concatenate(non_empty, axis=0)

        self.t2_model.fit(all_windows)
        self.ae_model = train_autoencoder(
            all_windows,
            epochs=ae_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        self.gate = train_gate(
            t2_model=self.t2_model,
            ae_model=self.ae_model,
            windows=all_windows,
            epochs=gate_epochs,
            batch_size=batch_size,
            lr=lr,
            margin=margin,
            anom_frac=anom_frac,
            seed=seed,
            device=device,
        )

        sample_fused_scores = []
        for w in sample_windows:
            if w.shape[0] == 0:
                sample_fused_scores.append(np.zeros((0,), dtype=np.float32))
                continue
            t2_s = self.t2_model.score(w)
            ae_s = score_autoencoder(self.ae_model, w, device=device)
            fused, _ = self.gate.fuse(w, t2_s, ae_s, device=device)
            sample_fused_scores.append(fused)

        self.ewma = EWMACalibrator(alpha=self.alpha)
        self.ewma.fit(sample_fused_scores)
        return self

    def score_sample(
        self,
        records: List[Dict],
        sample_id: str = "sample",
        device: str = "cpu",
    ) -> Dict:
        if self.ae_model is None or self.gate is None:
            raise RuntimeError("GVADDetector must be fitted or loaded before scoring.")

        raw, chroms, pos = extract_raw_features(records)
        norm_feat = self.scaler.transform(raw)
        windows, meta = build_windows(
            norm_feat,
            chroms,
            pos,
            window_size=self.window_size,
            stride=self.stride,
        )

        t2_scores = self.t2_model.score(windows)
        ae_scores = score_autoencoder(self.ae_model, windows, device=device)
        fused_scores, weights = self.gate.fuse(windows, t2_scores, ae_scores, device=device)
        ewma_res = self.ewma.evaluate_sample(fused_scores)

        mean_contrib, disp_contrib = self.t2_model.decompose(windows)
        if windows.shape[0] > 0:
            mean_c = np.mean(mean_contrib, axis=0)
            disp_c = np.mean(disp_contrib, axis=0)
            contrib_summary = {
                name: {
                    "mean_shift": float(mean_c[i]),
                    "dispersion": float(disp_c[i]),
                }
                for i, name in enumerate(FEATURE_NAMES)
            }
        else:
            contrib_summary = {
                name: {"mean_shift": 0.0, "dispersion": 0.0}
                for name in FEATURE_NAMES
            }

        breach_idx = ewma_res["breach_idx"]
        breach_region = meta[breach_idx] if breach_idx is not None and breach_idx < len(meta) else None

        return {
            "sample_id": sample_id,
            "num_variants": len(records),
            "num_windows": int(windows.shape[0]),
            "q_max": ewma_res["q_max"],
            "alert": ewma_res["alert"],
            "breach_idx": breach_idx,
            "breach_region": breach_region,
            "q_traj": ewma_res["q_traj"],
            "t2_scores": t2_scores,
            "ae_scores": ae_scores,
            "fused_scores": fused_scores,
            "gate_weights": weights,
            "window_meta": meta,
            "feature_contributions": contrib_summary,
        }

    def save(self, path: str | Path) -> None:
        if self.ae_model is None or self.gate is None:
            raise RuntimeError("Cannot save unfitted GVADDetector.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window_size": self.window_size,
            "stride": self.stride,
            "alpha": self.alpha,
            "scaler": self.scaler.state_dict(),
            "t2": self.t2_model.state_dict(),
            "ae": self.ae_model.state_dict(),
            "gate": self.gate.state_dict(),
            "ewma": self.ewma.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "GVADDetector":
        payload = torch.load(path, map_location=device, weights_only=False)
        ae_model = Conv1DAutoencoder(
            in_channels=len(FEATURE_NAMES),
            window_size=int(payload["window_size"]),
            latent_dim=16,
        )
        ae_model.load_state_dict(payload["ae"])
        ae_model.eval()

        return cls(
            window_size=int(payload["window_size"]),
            stride=int(payload["stride"]),
            alpha=float(payload["alpha"]),
            scaler=FeatureScaler.from_state_dict(payload["scaler"]),
            t2_model=HotellingT2.from_state_dict(payload["t2"]),
            ae_model=ae_model,
            gate=GateFusion.from_state_dict(payload["gate"]),
            ewma=EWMACalibrator.from_state_dict(payload["ewma"]),
        )


def export_flagged_bed(result: Dict, warn_limit: float, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_traj = result["q_traj"]
    meta = result["window_meta"]
    with open(output_path, "w", encoding="utf-8") as f:
        for i, m in enumerate(meta):
            if q_traj[i] >= warn_limit:
                f.write(
                    f"{m['chrom']}\t{max(0, m['start_pos'] - 1)}\t{m['end_pos']}\t"
                    f"{result['sample_id']}_w{i}\t{q_traj[i]:.4f}\n"
                )


def annotate_vcf_filter(
    input_vcf: str | Path,
    result: Dict,
    warn_limit: float,
    crit_limit: float,
    output_vcf: str | Path,
) -> None:
    input_vcf = Path(input_vcf)
    output_vcf = Path(output_vcf)
    output_vcf.parent.mkdir(parents=True, exist_ok=True)

    flagged_intervals: Dict[str, List[tuple]] = {}
    q_traj = result["q_traj"]
    for i, m in enumerate(result["window_meta"]):
        if q_traj[i] >= warn_limit:
            tag = "GVAD_CRIT" if q_traj[i] >= crit_limit else "GVAD_WARN"
            flagged_intervals.setdefault(m["chrom"], []).append(
                (m["start_pos"], m["end_pos"], tag)
            )

    opener = gzip.open if input_vcf.suffix == ".gz" else open
    header_added = False
    with opener(input_vcf, "rt", encoding="utf-8", errors="ignore") as fin, open(
        output_vcf, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue
            if line.startswith("#CHROM"):
                if not header_added:
                    fout.write('##FILTER=<ID=GVAD_WARN,Description="GVAD EWMA warning region">\n')
                    fout.write('##FILTER=<ID=GVAD_CRIT,Description="GVAD EWMA critical region">\n')
                    header_added = True
                fout.write(line)
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 8:
                fout.write(line)
                continue
            chrom = parts[0]
            try:
                pos = int(parts[1])
            except ValueError:
                fout.write(line)
                continue

            tags = set()
            for s_pos, e_pos, tag in flagged_intervals.get(chrom, []):
                if s_pos <= pos <= e_pos:
                    tags.add(tag)

            if tags:
                existing = [] if parts[6] in {"PASS", "."} else parts[6].split(";")
                merged = sorted(set(existing) | tags)
                parts[6] = ";".join(merged)
            fout.write("\t".join(parts) + "\n")
