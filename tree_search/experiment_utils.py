"""Dataset and run-summary helpers for reproducible experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATHS = {
    "demo_targets": PROJECT_ROOT / "data" / "demo_targets.csv",
    "uspto_190": PROJECT_ROOT / "data" / "uspto_190_targets.csv",
    "drug_hunter_172": PROJECT_ROOT / "data" / "drug_hunter_172.txt",
    "drugs_and_natural_products": PROJECT_ROOT / "data" / "drugs_and_natural_products.csv",
}


def resolve_targets_path(dataset_name: str, targets_file: Optional[str] = None) -> Path:
    """Resolve the source file for a dataset or an explicit targets file.

    Any dataset can be swapped out by passing ``--targets_file`` with a custom ``.csv`` or ``.txt``.
    """
    if targets_file:
        path = Path(targets_file).expanduser()
    elif dataset_name in DEFAULT_DATASET_PATHS:
        path = DEFAULT_DATASET_PATHS[dataset_name]
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {', '.join(sorted(DEFAULT_DATASET_PATHS))}. "
            f"Pass --targets_file to use a custom dataset."
        )

    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_targets(dataset_name: str, targets_file: Optional[str] = None) -> list[str]:
    """Load target SMILES from TXT or CSV sources."""
    path = resolve_targets_path(dataset_name=dataset_name, targets_file=targets_file)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.reader(handle) if row]

        if not rows:
            return []

        smiles_index = next(
            (
                index
                for index, value in enumerate(rows[0])
                if value.strip().lower() == "smiles"
            ),
            None,
        )
        start_row = 1 if smiles_index is not None else 0
        index = 0 if smiles_index is None else smiles_index
        targets = []
        for row in rows[start_row:]:
            if index < len(row) and row[index].strip():
                targets.append(row[index].strip())
        return targets

    raise ValueError(f"Unsupported targets file format: {path}")


def write_run_summary(output_folder: str | Path, summary: dict) -> None:
    """Write a compact JSON summary for a batch experiment."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4, ensure_ascii=False)
