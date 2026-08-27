"""Dataset loading and run bookkeeping shared by the batch runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATHS = {
    "demo_targets": PROJECT_ROOT / "data" / "demo_targets.csv",
    "uspto_190": PROJECT_ROOT / "data" / "uspto_190_targets.csv",
    "drug_hunter_172": PROJECT_ROOT / "data" / "drug_hunter_172.txt",
    "drugs_and_natural_products": PROJECT_ROOT / "data" / "drugs_and_natural_products.csv",
}


def parse_optional_depth(value) -> Optional[int]:
    """Parse a depth limit; ``none`` means an unbounded search."""
    if value is None or str(value).strip().lower() in {"", "none"}:
        return None

    depth = int(value)
    if depth < 0:
        raise ValueError("max_depth must be non-negative or none")
    return depth


def resolve_targets_path(dataset_name: str, targets_file: Optional[str] = None) -> Path:
    """Resolve a bundled dataset name or an explicit ``.csv``/``.txt`` targets file."""
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
    return path


def load_targets(dataset_name: str, targets_file: Optional[str] = None) -> list[str]:
    """Load target SMILES from a TXT file (one per line) or a CSV column."""
    path = resolve_targets_path(dataset_name, targets_file)

    if path.suffix.lower() == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        return []

    column = next(
        (i for i, value in enumerate(rows[0]) if value.strip().lower() == "smiles"),
        None,
    )
    body = rows[1:] if column is not None else rows
    column = column or 0
    return [row[column].strip() for row in body if column < len(row) and row[column].strip()]


def write_json(path: str | Path, value: Any) -> None:
    """Write ``value`` as indented JSON, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4, ensure_ascii=False)


def target_seed(run_seed: int, target_index: int) -> int:
    """Derive a reproducible per-target seed from the run seed."""
    return int(np.random.SeedSequence([int(run_seed), int(target_index)]).generate_state(1)[0])


def target_record_path(output_dir: str | Path, target_index: int) -> Path:
    return Path(output_dir) / "targets" / f"{target_index:06d}.json"


def read_target_record(path: str | Path) -> Optional[dict]:
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete(path: str | Path) -> bool:
    record = read_target_record(path)
    return bool(record) and record.get("status") == "complete"


def summarize_run(output_dir: str | Path, target_count: int) -> dict[str, Any]:
    """Aggregate the per-target records and write ``run_summary.json``."""
    output_dir = Path(output_dir)
    summary = {
        "target_count": int(target_count),
        "completed_count": 0,
        "success_count": 0,
        "error_count": 0,
        "missing_count": 0,
        "total_model_calls": 0,
    }
    for index in range(int(target_count)):
        record = read_target_record(target_record_path(output_dir, index))
        if record is None:
            summary["missing_count"] += 1
        elif record.get("status") == "complete":
            summary["completed_count"] += 1
            summary["success_count"] += int(bool(record.get("solved")))
            summary["total_model_calls"] += int(record.get("model_calls", 0))
        else:
            summary["error_count"] += 1

    write_json(output_dir / "run_summary.json", summary)
    return summary
