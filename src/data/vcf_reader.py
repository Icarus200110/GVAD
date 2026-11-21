import csv
import gzip
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


CHROM_ORDER: Dict[str, int] = {
    **{str(i): i for i in range(1, 23)},
    **{f"chr{i}": i for i in range(1, 23)},
    "X": 23,
    "chrX": 23,
    "Y": 24,
    "chrY": 24,
    "M": 25,
    "MT": 25,
    "chrM": 25,
    "chrMT": 25,
}


def chrom_sort_key(chrom: str) -> tuple:
    if chrom in CHROM_ORDER:
        return (0, CHROM_ORDER[chrom], chrom)
    return (1, 0, chrom)


def _safe_float(value: Optional[str], default: float = np.nan) -> float:
    if value is None or value == "." or value == "":
        return default
    if "," in value:
        value = value.split(",")[0]
    upper = value.strip().upper()
    if upper in {"INF", "+INF", "INFINITY"}:
        return np.inf
    if upper in {"-INF", "-INFINITY"}:
        return -np.inf
    try:
        return float(value)
    except ValueError:
        return default


def _parse_info(info_str: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not info_str or info_str == ".":
        return parsed
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            parsed[k] = v
        else:
            parsed[item] = "1"
    return parsed


def _parse_format(fmt_str: str, sample_str: str) -> Dict[str, str]:
    if not fmt_str or not sample_str or fmt_str == "." or sample_str == ".":
        return {}
    keys = fmt_str.split(":")
    vals = sample_str.split(":")
    return {k: vals[i] if i < len(vals) else "." for i, k in enumerate(keys)}


def read_vcf(path: str | Path, sample_index: int = 0) -> List[Dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records: List[Dict] = []

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 8:
                continue

            chrom = parts[0]
            try:
                pos = int(parts[1])
            except ValueError:
                continue
            qual = _safe_float(parts[5], default=0.0)
            flt = parts[6]
            info = _parse_info(parts[7])

            fmt_map: Dict[str, str] = {}
            if len(parts) >= 10:
                col_idx = 9 + sample_index
                if col_idx < len(parts):
                    fmt_map = _parse_format(parts[8], parts[col_idx])

            def get_metric(key: str, default: float = 0.0) -> float:
                if key in fmt_map:
                    v = _safe_float(fmt_map[key], default=np.nan)
                    if not np.isnan(v):
                        return v
                if key in info:
                    v = _safe_float(info[key], default=np.nan)
                    if not np.isnan(v):
                        return v
                return default

            dp = get_metric("DP", 1.0)
            mq = get_metric("MQ", 0.0)
            sor = get_metric("SOR", 0.0)
            af = get_metric("AF", 0.0)
            hiaf = get_metric("HIAF", af)
            pmean = get_metric("PMEAN", 0.0)
            pstd = get_metric("PSTD", 0.0)
            qd = get_metric("QD", qual / max(dp, 1.0))

            records.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "filter": flt,
                    "QUAL": qual,
                    "MQ": mq,
                    "DP": dp,
                    "QD": qd,
                    "SOR": sor,
                    "AF": af,
                    "HIAF": hiaf,
                    "PMEAN": pmean,
                    "PSTD": pstd,
                }
            )

    records.sort(key=lambda r: (chrom_sort_key(r["chrom"]), r["pos"]))
    return records


def read_tabular(path: str | Path) -> List[Dict]:
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    records: List[Dict] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            chrom = row.get("chrom") or row.get("CHROM") or "chr1"
            pos = int(float(row.get("pos") or row.get("POS") or 0))
            qual = _safe_float(row.get("QUAL"), 0.0)
            dp = _safe_float(row.get("DP"), 1.0)
            records.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "filter": row.get("filter") or row.get("FILTER") or "PASS",
                    "QUAL": qual,
                    "MQ": _safe_float(row.get("MQ"), 0.0),
                    "DP": dp,
                    "QD": _safe_float(row.get("QD"), qual / max(dp, 1.0)),
                    "SOR": _safe_float(row.get("SOR"), 0.0),
                    "AF": _safe_float(row.get("AF"), 0.0),
                    "HIAF": _safe_float(row.get("HIAF"), _safe_float(row.get("AF"), 0.0)),
                    "PMEAN": _safe_float(row.get("PMEAN"), 0.0),
                    "PSTD": _safe_float(row.get("PSTD"), 0.0),
                }
            )

    records.sort(key=lambda r: (chrom_sort_key(r["chrom"]), r["pos"]))
    return records


def load_variant_file(path: str | Path, sample_index: int = 0) -> List[Dict]:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return read_vcf(path, sample_index=sample_index)
    return read_tabular(path)


def load_directory(input_dir: str | Path, sample_index: int = 0) -> List[Dict]:
    input_dir = Path(input_dir)
    patterns = ["*.vcf", "*.vcf.gz", "*.csv", "*.tsv"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(input_dir.glob(pat)))

    samples: List[Dict] = []
    for fp in files:
        records = load_variant_file(fp, sample_index=sample_index)
        if records:
            samples.append(
                {
                    "sample_id": fp.stem.replace(".vcf", ""),
                    "path": str(fp),
                    "records": records,
                }
            )
    return samples
