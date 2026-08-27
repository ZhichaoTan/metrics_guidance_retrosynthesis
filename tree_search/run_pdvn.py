"""Run the unguided PDVN-SPRS policy with zero-value Retro* search."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tree_search.binghong_mlp_one_step import (
    BinghongStock,
    canonicalize_smiles,
    default_binghong_root,
)
from tree_search.experiment_utils import (
    is_complete,
    load_targets,
    parse_optional_depth,
    summarize_run,
    target_record_path,
    target_seed,
    write_json,
)
from tree_search.pdvn.one_step import PDVNSPRSOneStep
from tree_search.pdvn.planner import PDVNRetroStarPlanner

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--targets-file", "--targets_file")
    parser.add_argument("--binghong-root", "--binghong_root")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--reference-checkpoint", "--reference_checkpoint"
    )
    parser.add_argument("--templates-path", "--templates_path")
    parser.add_argument("--buyables-path", "--buyables_path")
    parser.add_argument(
        "--expansion-topk", "--expansion_topk", type=int, default=50
    )
    parser.add_argument(
        "--max-model-calls", "--max_model_calls", type=int, default=500
    )
    parser.add_argument(
        "--max-depth",
        "--max_depth",
        type=parse_optional_depth,
        default=10,
        help="Maximum search depth; use none for an unbounded search.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument(
        "--resume-existing", "--resume_existing", action="store_true"
    )
    return parser.parse_args(argv)


def _resolve_assets(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.binghong_root or default_binghong_root()).expanduser().resolve()
    one_step_root = root / "retro_star" / "one_step_model"
    return {
        "binghong_root": root,
        "checkpoint": Path(
            args.checkpoint
            or PROJECT_ROOT / "tree_search" / "pdvn" / "assets" / "rollout_model.ckpt"
        ).expanduser().resolve(),
        "reference": Path(
            args.reference_checkpoint
            or one_step_root / "saved_rollout_state_1_2048.ckpt"
        ).expanduser().resolve(),
        "templates": Path(
            args.templates_path or one_step_root / "template_rules_1.dat"
        ).expanduser().resolve(),
        "stock": Path(
            args.buyables_path or root / "retro_star" / "dataset" / "origin_dict.csv"
        ).expanduser().resolve(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    assets = _resolve_assets(args)
    targets = load_targets(args.dataset_name, args.targets_file)
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --gpu -1 for CPU mode")
        torch.cuda.set_device(args.gpu)

    one_step = PDVNSPRSOneStep(
        checkpoint=assets["checkpoint"],
        reference_checkpoint=assets["reference"],
        templates_path=assets["templates"],
        binghong_root=assets["binghong_root"],
        expansion_topk=args.expansion_topk,
        device=args.gpu,
    )
    planner = PDVNRetroStarPlanner(
        expand_fn=one_step,
        is_known_fn=BinghongStock(assets["stock"]),
        binghong_root=assets["binghong_root"],
        max_model_calls=args.max_model_calls,
        search_max_depth=args.max_depth,
    )

    for index, target in enumerate(tqdm(targets, desc="PDVN-SPRS + Retro*-0")):
        path = target_record_path(output_dir, index)
        if args.resume_existing and is_complete(path):
            continue
        seed = target_seed(args.seed, index)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        canonical_target = None
        try:
            canonical_target = canonicalize_smiles(target)
            result = planner.plan(canonical_target)
            if result.model_calls > args.max_model_calls:
                raise AssertionError("One-step call budget exceeded")
            if (
                result.route is not None
                and args.max_depth is not None
                and result.route["max_molecule_depth"] > args.max_depth
            ):
                raise AssertionError("Successful route exceeds the search depth")
            record = {
                "index": index,
                "target_smiles": target,
                "canonical_smiles": canonical_target,
                "seed": seed,
                "status": "complete",
                "solved": result.solved,
                "model_calls": result.model_calls,
                "termination": result.termination_reason,
                "route": result.route,
            }
        except Exception as error:
            record = {
                "index": index,
                "target_smiles": target,
                "canonical_smiles": canonical_target,
                "seed": seed,
                "status": "error",
                "solved": False,
                "model_calls": 0,
                "termination": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        write_json(path, record)

    summary = summarize_run(output_dir, len(targets))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
