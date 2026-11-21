# GVAD: Gated Multivariate Variant Anomaly Detection

Implementation of **GVAD: Gated Multivariate Variant Anomaly Detection for Real-Time Quality Control of Somatic Variant Calling**.

## Structure

```text
GVAD/
├── src/
│   ├── data/
│   │   ├── vcf_reader.py        # VarDict VCF / VCF.GZ / CSV / TSV reader
│   │   ├── preprocessing.py     # MAD screening, 10-feature extraction, sliding windows
│   │   └── synthesis.py         # Changepoint-persistent 6-mode anomaly synthesis
│   ├── models/
│   │   ├── hotelling.py         # Ledoit-Wolf Hotelling T^2 detector + decomposition
│   │   ├── autoencoder.py       # Conv1D autoencoder reconstruction detector
│   │   ├── gating.py            # 3-layer MLP gating fusion with ranking loss
│   │   ├── ewma.py              # Sample-level EWMA accumulation & threshold calibration
│   │   └── gvad.py              # End-to-end GVAD pipeline & BED/VCF output
│   └── evaluation/
│       ├── metrics.py           # ROC-AUC, PR-AUC, FPR operating points, severity/onset eval
│       └── make_paper_figures.py# ROC & EWMA trajectory figure plotting
├── scripts/
│   ├── train.py                 # Train GVAD baseline from normal VCF/CSV directory
│   ├── evaluate.py              # Evaluate on normal + synthetic anomaly samples
│   └── predict.py               # Score new VCF/CSV samples and export alerts/BED
├── paper/                       # LaTeX manuscript and figures
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

python scripts/train.py --input-dir path/to/normal_vcfs --output checkpoints/gvad.pt
python scripts/evaluate.py --input-dir path/to/normal_vcfs --checkpoint checkpoints/gvad.pt --output-dir outputs
python scripts/predict.py --input path/to/sample.vcf --checkpoint checkpoints/gvad.pt --output-dir outputs
```
