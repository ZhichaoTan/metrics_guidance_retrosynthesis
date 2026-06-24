"""Batch runner for metric-guided retrosynthetic search."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from copy import copy
from datetime import datetime

from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tree_search.experiment_utils import load_targets, resolve_targets_path, write_run_summary
from tree_search.mcts.metric_aggregation_utils import RANK_AGGREGATION_MODE
from tree_search.mcts.modules.local_pricer_api import preload_buyables, resolve_buyables_path

MODEL_PRIOR_PUCT_MODE = "model_prior_puct"
METRIC_HYBRID_PUCT_MODE = "metric_hybrid_puct"


class SearchTimeoutError(TimeoutError):
    """Raised when a single-target search exceeds its configured wall-clock budget."""


def _timeout_handler(signum, frame):
    raise SearchTimeoutError("Per-target search timed out.")


def _target_output_path(pathway_output_folder: str, target_smiles: str) -> str:
    prefix = target_smiles.replace("/", "_").replace("\\", "__")
    return os.path.join(pathway_output_folder, prefix)


def _write_target_outputs(output_path: str, paths, stats, graph) -> None:
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "paths.json"), "w", encoding="utf-8") as handle:
        json.dump(paths, handle, indent=4, ensure_ascii=False)
    with open(os.path.join(output_path, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=4, ensure_ascii=False)
    with open(os.path.join(output_path, "graph.json"), "w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=4, ensure_ascii=False)


def _write_failed_target_outputs(
    output_path: str,
    elapsed_seconds: float,
    *,
    error_message: str,
    timed_out: bool = False,
    timeout_seconds: int | None = None,
) -> None:
    stats = {
        "total_iterations": 0,
        "total_chemicals": 0,
        "total_reactions": 0,
        "total_templates": 0,
        "total_paths": 0,
        "first_path_time": None,
        "build_time": float(elapsed_seconds),
        "path_time": 0.0,
        "timed_out": bool(timed_out),
        "timeout_seconds": int(timeout_seconds) if timeout_seconds else None,
        "failed": True,
        "error_message": error_message,
    }
    _write_target_outputs(output_path, [], stats, {})


def setup_logging(log_file: str) -> logging.Logger:
    """Return a logger that writes to both ``log_file`` and the console."""
    logger = logging.getLogger("MetricGuidedRetrosynthesis")
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


def evaluate_search_success(target_smiles, model_name_list, args):
    """Run metric-guided MCTS for one target and write its route/graph/stats outputs.

    Returns ``(target_smiles, solved)`` where ``solved`` is True if a buyable route was found.
    A positive ``args.per_target_timeout`` arms a SIGALRM wall-clock guard so a single pathological
    target cannot stall the whole batch. It is a safety net, not a substitute for a bounded search.
    """
    search_start = time.time()
    output_path = _target_output_path(args.pathway_output_folder, target_smiles)
    if getattr(args, "buyables_path", None):
        os.environ["LOCAL_BUYABLES_PATH"] = os.path.abspath(args.buyables_path)

    timeout_seconds = int(getattr(args, "per_target_timeout", 0) or 0)
    timeout_enabled = timeout_seconds > 0 and hasattr(signal, "setitimer")
    previous_handler = None

    from tree_search.mcts.mcts_ged_controller import MCTSGED
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

        controller = MCTSGED(
            metric_name=str(args.metric_name),
            metric_weight_start=float(args.metric_weight_start),
            metric_weight_end=float(args.metric_weight_end),
            metric_schedule_type=str(args.metric_schedule_type),
            guidance_mode=str(args.guidance_mode),
            metric_aggregation_mode=str(args.metric_aggregation_mode),
            include_model_prior=bool(int(args.include_model_prior)),
        )

        build_tree_options = BuildTreeOptions()
        build_tree_options.max_iterations = int(args.max_iterations)
        build_tree_options.return_first = bool(int(args.return_first))
        build_tree_options.expansion_time = int(args.expansion_time)
        build_tree_options.max_depth = int(args.max_depth)

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
        return target_smiles, bool(stats["total_paths"] > 0)

    except SearchTimeoutError as exc:
        print(f"[Timeout] {target_smiles} exceeded {timeout_seconds}s", flush=True)
        _write_failed_target_outputs(
            output_path,
            time.time() - search_start,
            error_message=str(exc),
            timed_out=True,
            timeout_seconds=timeout_seconds,
        )
        return target_smiles, False
    except Exception as exc:
        # Isolate a single failed target so the batch continues; the traceback stays visible.
        traceback.print_exc()
        _write_failed_target_outputs(
            output_path,
            time.time() - search_start,
            error_message=str(exc),
            timeout_seconds=timeout_seconds,
        )
        return target_smiles, False
    finally:
        if timeout_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def _run_single_process(targets, logger, args) -> int:
    success_count = 0
    for target_smiles in tqdm(targets, total=len(targets), desc="Processing molecules"):
        _, success = evaluate_search_success(target_smiles, args.model_name_list, args)
        if success:
            success_count += 1
            logger.info("Pathway found for %s", target_smiles)
        else:
            logger.info("No pathway found for %s", target_smiles)
    return success_count


def _run_multi_process(targets, logger, args) -> int:
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                evaluate_search_success,
                target,
                args.model_name_list,
                copy(args),
            ): target
            for target in targets
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing molecules",
        ):
            target_smiles = futures[future]
            try:
                _, success = future.result()
            except Exception as exc:
                logger.error("Worker failed for %s: %s", target_smiles, exc)
                success = False
            if success:
                success_count += 1
                logger.info("Pathway found for %s", target_smiles)
            else:
                logger.info("No pathway found for %s", target_smiles)
    return success_count


def main(targets, logger, args):
    """Search every target (single- or multi-process) and return the run summary dict."""
    start_time = time.time()
    logger.info("Starting process for %d molecules...", len(targets))

    buyables_path = resolve_buyables_path(getattr(args, "buyables_path", None))
    start_method = mp.get_start_method(allow_none=True)
    if args.num_workers > 1 and start_method == "fork":
        preload_buyables(buyables_path)
        logger.info(
            "Preloaded buyables in parent process for fork-based worker sharing: %s",
            buyables_path,
        )

    if args.num_workers == 1:
        success_count = _run_single_process(targets, logger, args)
    else:
        success_count = _run_multi_process(targets, logger, args)

    elapsed_seconds = time.time() - start_time
    success_rate = (success_count / len(targets)) * 100 if targets else 0.0
    logger.info("Processed %d compounds.", len(targets))
    logger.info("Found valid pathways in %d compounds.", success_count)
    logger.info("Search success rate: %.2f%%", success_rate)
    logger.info("Total time taken: %.2f seconds", elapsed_seconds)
    return {
        "target_count": len(targets),
        "success_count": success_count,
        "failed_count": len(targets) - success_count,
        "success_rate": success_rate,
        "elapsed_seconds": elapsed_seconds,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ``run_mcts`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run metric-guided retrosynthetic search on a dataset."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--targets_file", type=str, default=None)
    parser.add_argument("--model_name_list", type=str, nargs="+", required=True)
    parser.add_argument("--log_output", type=str, default="log_output")
    parser.add_argument("--pathway_output_folder", type=str, default="pathway_output_folder")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--return_first", type=int, default=1)
    parser.add_argument("--expansion_time", type=int, default=12000)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--one_step_max_num_templates", type=int, default=25)
    parser.add_argument("--sorting_metric", type=str, default="number_of_reactions")
    parser.add_argument("--metric_name", type=str, default="ged")
    parser.add_argument("--guidance_mode", type=str, default=METRIC_HYBRID_PUCT_MODE)
    parser.add_argument(
        "--metric_aggregation_mode",
        type=str,
        default=RANK_AGGREGATION_MODE,
    )
    parser.add_argument("--include_model_prior", type=int, default=1)
    parser.add_argument("--metric_weight_start", type=float, default=0.0)
    parser.add_argument("--metric_weight_end", type=float, default=1.0)
    parser.add_argument("--metric_schedule_type", type=str, default="constant")
    parser.add_argument(
        "--retro_backend",
        type=str,
        default=os.environ.get("RETRO_ONE_STEP_BACKEND", "template_relevance"),
    )
    parser.add_argument("--buyables_path", type=str, default=None)
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
        f"_iter{args.max_iterations}_d{args.max_depth}_{timestamp}.log"
    )
    logger = setup_logging(os.path.join(args.log_output, log_name))

    targets_path = resolve_targets_path(args.dataset_name, args.targets_file)
    targets = load_targets(args.dataset_name, args.targets_file)
    logger.info("Loaded %d target molecules from %s", len(targets), targets_path)

    summary = main(targets, logger, args)
    summary.update(
        {
            "dataset_name": args.dataset_name,
            "targets_file": str(targets_path),
            "model_name_list": args.model_name_list,
            "num_workers": int(args.num_workers),
            "max_iterations": int(args.max_iterations),
            "max_depth": int(args.max_depth),
            "one_step_max_num_templates": int(args.one_step_max_num_templates),
            "return_first": bool(int(args.return_first)),
            "expansion_time": int(args.expansion_time),
            "sorting_metric": args.sorting_metric,
            "metric_name": args.metric_name,
            "guidance_mode": args.guidance_mode,
            "metric_aggregation_mode": args.metric_aggregation_mode,
            "include_model_prior": bool(int(args.include_model_prior)),
            "metric_weight_start": float(args.metric_weight_start),
            "metric_weight_end": float(args.metric_weight_end),
            "metric_schedule_type": args.metric_schedule_type,
            "buyables_path": resolve_buyables_path(args.buyables_path),
            "retro_backend": str(args.retro_backend),
            "per_target_timeout": int(args.per_target_timeout),
            "backend": "ged_mcts",
        }
    )
    write_run_summary(args.pathway_output_folder, summary)


if __name__ == "__main__":
    run_cli()
