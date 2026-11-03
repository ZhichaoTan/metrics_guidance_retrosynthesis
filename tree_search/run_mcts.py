"""Batch runner for MCTS retrosynthetic search with the complexity-distance guidance."""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from copy import copy
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tree_search.experiment_utils import (
    is_complete,
    load_targets,
    parse_optional_depth,
    summarize_run,
    target_record_path,
    target_seed,
    write_json,
)
from tree_search.mcts.guidance import METRIC_HYBRID_PUCT_MODE, RANK_AGGREGATION_MODE
from tree_search.mcts.modules.local_pricer_api import preload_buyables, resolve_buyables_path


class SearchTimeoutError(TimeoutError):
    """Raised when a single-target search exceeds its configured wall-clock budget."""


def _timeout_handler(signum, frame):
    raise SearchTimeoutError("Per-target search timed out.")


def _target_output_path(pathway_output_folder: str, target_smiles: str) -> str:
    prefix = target_smiles.replace("/", "_").replace("\\", "__")
    return os.path.join(pathway_output_folder, prefix)


def _set_target_seed(run_seed: int, target_index: int) -> int:
    import numpy as np
    import torch

    seed = target_seed(run_seed, target_index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _write_target_record(
    output_dir: str | Path,
    target_index: int,
    target_smiles: str,
    *,
    status: str,
    solved: bool,
    model_calls: int,
    termination: str,
    seed: int | None = None,
    error: str | None = None,
) -> None:
    record = {
        "index": int(target_index),
        "target_smiles": target_smiles,
        "status": status,
        "solved": bool(solved),
        "model_calls": int(model_calls),
        "termination": termination,
    }
    if seed is not None:
        record["seed"] = int(seed)
    if error is not None:
        record["error"] = error
    write_json(target_record_path(output_dir, target_index), record)


def _normal_termination(stats: dict, args) -> str:
    if stats["search_solved"]:
        return "solved"
    model_calls = int(stats.get("model_calls", stats.get("total_iterations", 0)))
    if model_calls >= int(args.max_iterations):
        return "model_call_budget"
    expansion_time = parse_optional_depth(args.expansion_time)
    if (
        expansion_time is not None
        and float(stats.get("build_time", 0.0)) >= expansion_time
    ):
        return "time_budget"
    return "search_exhausted"


def _pending_targets(targets, args):
    indexed_targets = list(enumerate(targets))
    if not args.resume_existing:
        return indexed_targets
    output_dir = Path(args.pathway_output_folder)
    return [
        (index, target)
        for index, target in indexed_targets
        if not is_complete(target_record_path(output_dir, index))
    ]


def _write_target_outputs(output_path: str, paths, stats, graph) -> None:
    for name, value in (("paths", paths), ("stats", stats), ("graph", graph)):
        write_json(os.path.join(output_path, f"{name}.json"), value)


def _write_failed_target_outputs(
    output_path: str,
    elapsed_seconds: float,
    *,
    error_message: str,
    model_calls: int = 0,
    timed_out: bool = False,
    timeout_seconds: int | None = None,
) -> None:
    stats = {
        "total_iterations": int(model_calls),
        "model_calls": int(model_calls),
        "total_chemicals": 0,
        "total_reactions": 0,
        "total_templates": 0,
        "total_paths": 0,
        "search_solved": False,
        "enumerate_solved": False,
        "search_succ": False,
        "enumerate_succ": False,
        "first_path_time": None,
        "build_time": float(elapsed_seconds),
        "path_time": 0.0,
        "timed_out": bool(timed_out),
        "timeout_seconds": int(timeout_seconds) if timeout_seconds else None,
        "failed": True,
        "error": error_message,
        "error_message": error_message,
    }
    _write_target_outputs(output_path, [], stats, {})


def setup_logging(log_file: str) -> logging.Logger:
    """Return a logger that writes to both ``log_file`` and the console."""
    logger = logging.getLogger("ComplexityDistanceGuidance")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _short_log_tag(value: str) -> str:
    return (
        str(value)
        .replace("sa_score", "sa")
        .replace("sc_score", "sc")
        .replace("tanimoto", "tan")
        .replace("number_of_reactions", "numrxn")
        .replace("number_of_starting_materials", "numstart")
        .replace("metric_hybrid_puct", "hybrid")
        .replace("model_prior_puct", "model-prior-puct")
        .replace("rank_aggregation", "rank")
        .replace("signed_rank_positive_only", "srpos")
        .replace("&", "+")
        .replace("_", "-")
    )


def evaluate_search_success(target_index, target_smiles, model_name_list, args):
    """Run MCTS for one target and write its route/graph/stats outputs.

    Returns ``(target_smiles, solved)`` using the search tree's solved flag.
    A positive ``args.per_target_timeout`` arms a SIGALRM wall-clock guard so a single pathological
    target cannot stall the whole batch. It is a safety net, not a substitute for a bounded search.
    """
    search_start = time.time()
    seed = _set_target_seed(args.seed, target_index)
    output_path = _target_output_path(args.pathway_output_folder, target_smiles)
    if getattr(args, "buyables_path", None):
        os.environ["LOCAL_BUYABLES_PATH"] = os.path.abspath(args.buyables_path)
    os.environ["BINGHONG_MLP_DEVICE"] = str(args.gpu)

    timeout_seconds = int(getattr(args, "per_target_timeout", 0) or 0)
    timeout_enabled = timeout_seconds > 0 and hasattr(signal, "setitimer")
    previous_handler = None
    controller = None

    from tree_search.mcts.complexity_distance_controller import ComplexityDistanceMCTS
    from tree_search.mcts.options import (
        BuildTreeOptions,
        EnumeratePathsOptions,
        ExpandOneOptions,
        RetroBackendOption,
    )

    try:
        if timeout_enabled:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

        controller = ComplexityDistanceMCTS(
            metric_name=str(args.metric_name),
            metric_weight_start=float(args.metric_weight_start),
            metric_weight_end=float(args.metric_weight_end),
            metric_schedule_type=str(args.metric_schedule_type),
            guidance_mode=str(args.guidance_mode),
            metric_aggregation_mode=str(args.metric_aggregation_mode),
        )

        build_tree_options = BuildTreeOptions()
        build_tree_options.max_iterations = int(args.max_iterations)
        build_tree_options.return_first = bool(int(args.return_first))
        build_tree_options.expansion_time = parse_optional_depth(args.expansion_time)
        build_tree_options.max_depth = parse_optional_depth(args.max_depth)

        expand_one_options = ExpandOneOptions()
        expand_one_options.retro_backend_options = []
        retro_backend = str(args.retro_backend)
        for model_name in model_name_list:
            option = RetroBackendOption()
            option.retro_backend = retro_backend
            option.retro_model_name = model_name
            option.max_num_templates = int(args.one_step_max_num_templates)
            expand_one_options.retro_backend_options.append(option)

        paths, stats, graph = controller.get_buyable_paths(
            target=target_smiles,
            expand_one_options=expand_one_options,
            build_tree_options=build_tree_options,
            enumerate_paths_options=EnumeratePathsOptions(
                sorting_metric=str(args.sorting_metric)
            ),
        )
        _write_target_outputs(output_path, paths, stats, graph)
        solved = bool(stats["search_solved"])
        _write_target_record(
            args.pathway_output_folder,
            target_index,
            target_smiles,
            status="complete",
            solved=solved,
            model_calls=int(stats["model_calls"]),
            termination=_normal_termination(stats, args),
            seed=seed,
        )
        return target_smiles, solved

    except SearchTimeoutError as exc:
        model_calls = int(getattr(controller, "iterations", 0))
        print(f"[Timeout] {target_smiles} exceeded {timeout_seconds}s", flush=True)
        _write_failed_target_outputs(
            output_path,
            time.time() - search_start,
            error_message=str(exc),
            model_calls=model_calls,
            timed_out=True,
            timeout_seconds=timeout_seconds,
        )
        _write_target_record(
            args.pathway_output_folder,
            target_index,
            target_smiles,
            status="error",
            solved=False,
            model_calls=model_calls,
            termination="timeout",
            seed=seed,
            error=str(exc),
        )
        return target_smiles, False
    except Exception as exc:
        model_calls = int(getattr(controller, "iterations", 0))
        # Isolate a single failed target so the batch continues; the traceback stays visible.
        traceback.print_exc()
        _write_failed_target_outputs(
            output_path,
            time.time() - search_start,
            error_message=str(exc),
            model_calls=model_calls,
            timeout_seconds=timeout_seconds,
        )
        _write_target_record(
            args.pathway_output_folder,
            target_index,
            target_smiles,
            status="error",
            solved=False,
            model_calls=model_calls,
            termination="error",
            seed=seed,
            error=str(exc),
        )
        return target_smiles, False
    finally:
        if timeout_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def _run_single_process(indexed_targets, logger, args) -> None:
    for target_index, target_smiles in tqdm(
        indexed_targets,
        total=len(indexed_targets),
        desc="Processing molecules",
    ):
        _, success = evaluate_search_success(
            target_index,
            target_smiles,
            args.model_name_list,
            args,
        )
        if success:
            logger.info("Pathway found for %s", target_smiles)
        else:
            logger.info("No pathway found for %s", target_smiles)


def _run_multi_process(indexed_targets, logger, args) -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                evaluate_search_success,
                target_index,
                target,
                args.model_name_list,
                copy(args),
            ): (target_index, target)
            for target_index, target in indexed_targets
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing molecules",
        ):
            target_index, target_smiles = futures[future]
            try:
                _, success = future.result()
            except Exception as exc:
                logger.error("Worker failed for %s: %s", target_smiles, exc)
                message = f"Worker failed: {exc}"
                _write_failed_target_outputs(
                    _target_output_path(args.pathway_output_folder, target_smiles),
                    0.0,
                    error_message=message,
                    timeout_seconds=int(args.per_target_timeout),
                )
                _write_target_record(
                    args.pathway_output_folder,
                    target_index,
                    target_smiles,
                    status="error",
                    solved=False,
                    model_calls=0,
                    termination="worker_error",
                    seed=target_seed(args.seed, target_index),
                    error=message,
                )
                success = False
            if success:
                logger.info("Pathway found for %s", target_smiles)
            else:
                logger.info("No pathway found for %s", target_smiles)


def main(targets, logger, args):
    """Search every target (single- or multi-process) and return the run summary dict."""
    start_time = time.time()
    indexed_targets = _pending_targets(targets, args)
    resumed_count = len(targets) - len(indexed_targets)
    logger.info(
        "Starting process for %d molecules (%d resumed).",
        len(indexed_targets),
        resumed_count,
    )

    buyables_path = resolve_buyables_path(getattr(args, "buyables_path", None))
    start_method = mp.get_start_method(allow_none=True)
    if indexed_targets and args.num_workers > 1 and start_method == "fork":
        preload_buyables(buyables_path)
        logger.info(
            "Preloaded buyables in parent process for fork-based worker sharing: %s",
            buyables_path,
        )

    if args.num_workers == 1:
        _run_single_process(indexed_targets, logger, args)
    else:
        _run_multi_process(indexed_targets, logger, args)

    elapsed_seconds = time.time() - start_time
    summary = summarize_run(args.pathway_output_folder, len(targets))
    success_rate = (
        summary["success_count"] / len(targets) * 100 if targets else 0.0
    )
    logger.info("Completed %d/%d compounds.", summary["completed_count"], len(targets))
    logger.info("Found valid pathways in %d compounds.", summary["success_count"])
    logger.info("Search success rate: %.2f%%", success_rate)
    logger.info(
        "Errors: %d; missing: %d.",
        summary["error_count"],
        summary["missing_count"],
    )
    logger.info("Total time taken: %.2f seconds", elapsed_seconds)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the ``run_mcts`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run retrosynthetic search with the complexity-distance guidance on a dataset."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--targets_file", type=str, default=None)
    parser.add_argument("--model_name_list", type=str, nargs="+", required=True)
    parser.add_argument("--log_output", type=str, default="log_output")
    parser.add_argument("--pathway_output_folder", type=str, default="pathway_output_folder")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--return_first", type=int, default=1)
    parser.add_argument(
        "--expansion_time",
        type=parse_optional_depth,
        default=12000,
        help="Search wall-clock limit in seconds; use none to disable it.",
    )
    parser.add_argument(
        "--max_depth",
        type=parse_optional_depth,
        default=10,
        help="Maximum search depth; use none for an unbounded search.",
    )
    parser.add_argument("--one_step_max_num_templates", type=int, default=25)
    parser.add_argument("--sorting_metric", type=str, default="number_of_reactions")
    parser.add_argument("--metric_name", type=str, default="ged")
    parser.add_argument("--guidance_mode", type=str, default=METRIC_HYBRID_PUCT_MODE)
    parser.add_argument(
        "--metric_aggregation_mode",
        type=str,
        default=RANK_AGGREGATION_MODE,
    )
    parser.add_argument(
        "--include_model_prior",
        type=int,
        default=1,
        help="No effect on the search; kept for the log file name.",
    )
    parser.add_argument("--metric_weight_start", type=float, default=0.0)
    parser.add_argument("--metric_weight_end", type=float, default=1.0)
    parser.add_argument("--metric_schedule_type", type=str, default="constant")
    parser.add_argument(
        "--retro_backend",
        type=str,
        default=os.environ.get("RETRO_ONE_STEP_BACKEND", "template_relevance"),
        choices=("template_relevance", "binghong_mlp"),
    )
    parser.add_argument("--buyables_path", type=str, default=None)
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Binghong MLP GPU index; -1 selects CPU.",
    )
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="Skip targets that already have a complete record in the output folder.",
    )
    parser.add_argument(
        "--per_target_timeout",
        type=int,
        default=0,
        help="Wall-clock timeout in seconds for a single target search; 0 disables it.",
    )
    return parser


def run_cli(argv=None) -> None:
    """Parse args, run the batch search, and write ``run_summary.json``. CLI entry point."""
    args = build_parser().parse_args(argv)

    os.makedirs(args.log_output, exist_ok=True)
    os.makedirs(args.pathway_output_folder, exist_ok=True)
    if args.buyables_path:
        os.environ["LOCAL_BUYABLES_PATH"] = os.path.abspath(args.buyables_path)
    os.environ["BINGHONG_MLP_DEVICE"] = str(args.gpu)
    if args.gpu >= 0:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --gpu -1 for CPU mode")
        torch.cuda.set_device(args.gpu)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_name = "_".join(args.model_name_list)
    metric_tag = _short_log_tag(args.metric_name)
    guidance_tag = _short_log_tag(args.guidance_mode)
    aggregation_tag = _short_log_tag(args.metric_aggregation_mode)
    sorting_tag = _short_log_tag(args.sorting_metric)
    log_name = (
        f"{args.dataset_name}_{args.model_name}_m-{metric_tag}"
        f"_g-{guidance_tag}_a-{aggregation_tag}_mp-{args.include_model_prior}"
        f"_s-{sorting_tag}_{args.metric_schedule_type}"
        f"_iter{args.max_iterations}_d{args.max_depth or 'none'}_{timestamp}.log"
    )
    logger = setup_logging(os.path.join(args.log_output, log_name))

    targets = load_targets(args.dataset_name, args.targets_file)
    logger.info("Loaded %d target molecules for dataset %s", len(targets), args.dataset_name)
    main(targets, logger, args)


if __name__ == "__main__":
    run_cli()
