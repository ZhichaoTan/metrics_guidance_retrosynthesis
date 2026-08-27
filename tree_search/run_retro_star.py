"""Batch runner for Retro* search, optionally with planner-native rank fusion."""
import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_binghong_root = os.environ.get(
    "BINGHONG_RETRO_STAR_ROOT",
    os.path.join(_project_root, "external", "binghong_retro_star"),
)
if os.path.isdir(_binghong_root) and _binghong_root not in sys.path:
    sys.path.insert(0, _binghong_root)

from tree_search.experiment_utils import (
    is_complete,
    load_targets,
    parse_optional_depth,
    summarize_run,
    target_record_path,
    target_seed,
    write_json,
)
from tree_search.binghong_mlp_one_step import (
    canonicalize_smiles,
    ensure_binghong_on_path,
)
from tree_search.retro_star.adapters.depth_utils import (
    binghong_route_max_mol_depth,
)

_WORKER_PLANNER = None
_SHARED_STARTING_MOLECULES = None
_SHARED_STARTING_MOLECULES_PATH = None


def _import_rs_planner():
    import sys

    ensure_binghong_on_path(_binghong_root)
    argv = sys.argv
    sys.argv = ["binghong_retro_star"]
    try:
        from retro_star.api import RSPlanner
        return RSPlanner
    finally:
        sys.argv = argv


def _import_prepare_starting_molecules():
    import sys

    ensure_binghong_on_path(_binghong_root)
    argv = sys.argv
    sys.argv = ["binghong_retro_star"]
    try:
        from retro_star.common.prepare_utils import prepare_starting_molecules
        return prepare_starting_molecules
    finally:
        sys.argv = argv


def _preload_starting_molecules(starting_molecules_path, logger=None):
    """Preload starting molecules in parent for fork copy-on-write sharing."""
    global _SHARED_STARTING_MOLECULES, _SHARED_STARTING_MOLECULES_PATH

    resolved_path = os.path.abspath(starting_molecules_path)
    if (
        _SHARED_STARTING_MOLECULES is not None
        and _SHARED_STARTING_MOLECULES_PATH == resolved_path
    ):
        return _SHARED_STARTING_MOLECULES

    prepare_starting_molecules = _import_prepare_starting_molecules()
    _SHARED_STARTING_MOLECULES = prepare_starting_molecules(resolved_path)
    _SHARED_STARTING_MOLECULES_PATH = resolved_path
    if logger is not None:
        logger.info(
            "Preloaded starting molecules in parent process for fork-based "
            "worker sharing: %s (%d molecules)",
            resolved_path,
            len(_SHARED_STARTING_MOLECULES),
        )
    return _SHARED_STARTING_MOLECULES


def _patch_rsplanner_starting_molecules_loader(starting_molecules_path):
    """Make RSPlanner reuse the fork-shared starting molecule object when possible."""
    if _SHARED_STARTING_MOLECULES is None:
        return

    import retro_star.api as api_module

    original_loader = api_module.prepare_starting_molecules
    resolved_path = os.path.abspath(starting_molecules_path)

    def shared_loader(filename):
        if os.path.abspath(filename) == resolved_path:
            logging.info(
                "Using preloaded starting molecules from parent process: %s",
                resolved_path,
            )
            return _SHARED_STARTING_MOLECULES
        return original_loader(filename)

    api_module.prepare_starting_molecules = shared_loader


def setup_logging(log_file):
    """Return a logger that writes to both ``log_file`` and the console."""
    logger = logging.getLogger("BinghongRetroStarLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def _write_target_record(target_index, target_smiles, args, stats, route=None):
    record = {
        "index": int(target_index),
        "target_smiles": target_smiles,
        "seed": target_seed(args.seed, target_index),
        "status": stats["status"],
        "solved": bool(stats["search_succ"]),
        "search_succ": bool(stats["search_succ"]),
        "failed": bool(stats["failed"]),
        "termination": stats["termination"],
        "iterations": int(stats["iterations"]),
        "model_calls": int(stats["model_calls"]),
        "guidance_diagnostics": stats.get("guidance_diagnostics", {}),
    }
    if route is not None:
        record["route"] = route
    if stats.get("error") is not None:
        record["error"] = stats["error"]
    write_json(
        target_record_path(args.pathway_output_folder, target_index), record
    )


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


def _init_worker(
    gpu,
    use_value_fn,
    iterations,
    expansion_topk,
    max_depth,
    starting_molecules,
    one_step_backend,
    retro_model_name,
    guidance,
):
    global _WORKER_PLANNER
    RSPlanner = _import_rs_planner()
    _patch_rsplanner_starting_molecules_loader(starting_molecules)
    # binghong's parse_args (run at import, with scrubbed argv) force-sets
    # CUDA_VISIBLE_DEVICES from its OWN default (-1), hiding all GPUs. Restore the
    # requested GPU here -- in the worker, post-fork, before any CUDA context
    # is created -- so the value network and the one-step ("auto") model land
    # on the GPU. Each worker builds its own CUDA context after fork, which is
    # the fork-safe pattern.
    if int(gpu) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    _WORKER_PLANNER = RSPlanner(
        gpu=int(gpu),
        use_value_fn=bool(int(use_value_fn)),
        iterations=int(iterations),
        expansion_topk=int(expansion_topk),
        search_max_depth=max_depth,
        starting_molecules=starting_molecules,
        one_step_backend=one_step_backend,
        retro_model_name=retro_model_name,
        guidance=str(guidance),
    )


def _target_output_path(target_smiles, output_folder):
    prefix = target_smiles.replace("/", "_").replace("\\", "__")
    return os.path.join(output_folder, prefix)


def _write_target_artifacts(target_smiles, args, stats, paths):
    output_path = _target_output_path(target_smiles, args.pathway_output_folder)
    write_json(os.path.join(output_path, "paths.json"), paths)
    write_json(os.path.join(output_path, "stats.json"), stats)


def _error_stats(args, error, guidance_diagnostics=None, model_calls=0):
    return {
        "status": "error",
        "succ": False,
        "search_succ": False,
        "enumerate_succ": False,
        "failed": True,
        "error": f"{type(error).__name__}: {error}",
        "termination": "error",
        "iterations": 0,
        "model_calls": int(model_calls),
        "time": 0.0,
        "route_cost": None,
        "route_len": None,
        "route_max_mol_depth": None,
        "max_depth": args.max_depth,
        "guidance_diagnostics": guidance_diagnostics or {},
    }


def evaluate_search_success(target_index, target_smiles, args):
    """Plan one target and always write auditable route/stats artifacts."""
    import gc
    import traceback

    gc.collect()
    seed = target_seed(args.seed, target_index)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        canonical_target = canonicalize_smiles(target_smiles)
        result = _WORKER_PLANNER.plan(canonical_target)
        search_succ = bool(result and result.get("succ"))
        route_len_value = result.get("route_len") if result else None
        route_len = int(route_len_value) if route_len_value is not None else None
        route_str = result.get("routes") if result else None
        route_max_mol_depth = (
            binghong_route_max_mol_depth(route_str) if route_str else None
        )
        iterations = int(result.get("iter", 0)) if result else 0
        model_calls = int(result.get("model_calls", iterations)) if result else 0
        termination = (
            "solved"
            if search_succ
            else (
                "model_call_budget"
                if model_calls >= int(args.max_iterations)
                else "search_exhausted"
            )
        )

        stats = {
            "index": int(target_index),
            "status": "complete",
            "succ": search_succ,
            "search_succ": search_succ,
            "enumerate_succ": search_succ,
            "failed": False,
            "error": None,
            "termination": termination,
            "iterations": iterations,
            "model_calls": model_calls,
            "time": float(result.get("time", 0.0)) if result else 0.0,
            "route_cost": result.get("route_cost") if result else None,
            "route_len": route_len,
            "route_max_mol_depth": route_max_mol_depth,
            "max_depth": args.max_depth,
            "guidance_diagnostics": (
                result.get("guidance_diagnostics", {}) if result else {}
            ),
        }
        paths = []
        if search_succ and route_str is not None:
            paths = [{"binghong_route": result["routes"]}]
        _write_target_artifacts(target_smiles, args, stats, paths)
        _write_target_record(
            target_index,
            target_smiles,
            args,
            stats,
            route=route_str if search_succ else None,
        )

        gc.collect()
        return target_smiles, stats

    except Exception as e:
        print(f"[Error] Processing {target_smiles} failed: {e}")
        traceback.print_exc()
        diagnostics = {}
        if _WORKER_PLANNER is not None and hasattr(
            _WORKER_PLANNER, "guidance_diagnostics"
        ):
            diagnostics = _WORKER_PLANNER.guidance_diagnostics()
        stats = _error_stats(args, e, diagnostics)
        stats["index"] = int(target_index)
        _write_target_artifacts(target_smiles, args, stats, [])
        _write_target_record(target_index, target_smiles, args, stats)
        return target_smiles, stats


def main(targets, logger, args):
    """Plan every target across a worker pool and return the run summary dict."""
    start = time.time()
    pending_targets = _pending_targets(targets, args)
    if pending_targets:
        starting_molecules = os.path.abspath(args.buyables_path)
        if mp.get_start_method() == "fork":
            _preload_starting_molecules(starting_molecules, logger)
        else:
            logger.info(
                "Multiprocessing start method is %s; starting molecules will be "
                "loaded per worker.",
                mp.get_start_method(),
            )
        initargs = (
            int(args.gpu),
            int(args.use_value_network),
            int(args.max_iterations),
            int(args.expansion_topk),
            args.max_depth,
            starting_molecules,
            str(args.one_step_backend),
            str(args.retro_model_name),
            str(args.guidance),
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=initargs,
        ) as ex:
            futures = {
                ex.submit(evaluate_search_success, index, target, args): (
                    index,
                    target,
                )
                for index, target in pending_targets
            }
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="binghong Retro*",
            ):
                target_index, target = futures[fut]
                try:
                    target, stats = fut.result()
                    if stats["search_succ"]:
                        logger.info(
                            f"Pathway found for {target} "
                            f"(iter={stats.get('iterations', '?')}, "
                            f"time={stats.get('time', 0):.2f}s)"
                        )
                    else:
                        logger.info(f"No pathway found for {target}")
                except Exception as error:
                    stats = _error_stats(args, error)
                    stats["index"] = int(target_index)
                    _write_target_artifacts(target, args, stats, [])
                    _write_target_record(target_index, target, args, stats)
                    logger.error(f"Worker failed for {target}: {error}")

    target_count = len(targets)
    elapsed_seconds = time.time() - start
    summary = summarize_run(args.pathway_output_folder, target_count)
    logger.info(f"Processed {target_count} compounds in {elapsed_seconds:.2f}s.")
    logger.info(
        f"Search success (max_depth={args.max_depth}): "
        f"{summary['success_count']}/{target_count}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the ``run_retro_star`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Parallel binghong official Retro* (RSPlanner) search."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--log_output", type=str, default="log_output")
    parser.add_argument(
        "--pathway_output_folder",
        type=str,
        default="binghong_retro_star_outputs",
    )
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument(
        "--max_iterations",
        "--max_model_calls",
        dest="max_iterations",
        type=int,
        default=500,
    )
    parser.add_argument("--expansion_topk", type=int, default=50)
    parser.add_argument(
        "--buyables_path",
        type=str,
        default=None,
        help="Starting molecules file (Binghong CSV or local buyables.json.gz).",
    )
    parser.add_argument(
        "--one_step_backend",
        type=str,
        default="binghong_mlp",
        choices=["binghong_mlp", "template_relevance"],
        help="One-step model backend for RSPlanner.",
    )
    parser.add_argument(
        "--retro_model_name",
        type=str,
        default="uspto_original_consol_Roh",
        help="Template relevance model name when one_step_backend=template_relevance.",
    )
    parser.add_argument(
        "--max_depth",
        type=parse_optional_depth,
        default=10,
        help="Maximum search depth; use none for an unbounded search.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--use_value_network", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--targets_file", type=str, default=None)
    parser.add_argument(
        "--guidance",
        choices=["none", "planner_native_rank_fusion"],
        default="none",
        help=(
            "Optional fixed SA+SC+GED guidance applied only to Retro* "
            "frontier selection."
        ),
    )
    return parser


def run_cli(argv=None) -> int:
    """Run the canonical Retro* batch protocol and return its exit status."""
    args = build_parser().parse_args(argv)

    if args.buyables_path is None:
        args.buyables_path = os.path.join(
            _binghong_root,
            "retro_star",
            "dataset",
            "origin_dict.csv",
        )

    os.makedirs(args.log_output, exist_ok=True)
    os.makedirs(args.pathway_output_folder, exist_ok=True)

    depth_tag = "unbounded" if args.max_depth is None else f"d{args.max_depth}"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(
        args.log_output,
        f"binghong_retro_star_{args.dataset_name}_vn{args.use_value_network}_"
        f"topk{args.expansion_topk}_iter{args.max_iterations}_{depth_tag}_{ts}.log",
    )
    logger = setup_logging(log_file)

    targets = load_targets(args.dataset_name, args.targets_file)
    logger.info(
        f"Loaded {len(targets)} targets for dataset {args.dataset_name}; "
        f"binghong RSPlanner vn={args.use_value_network} "
        f"topk={args.expansion_topk} iter={args.max_iterations} "
        f"buyables={args.buyables_path} "
        f"one_step_backend={args.one_step_backend} "
        f"retro_model_name={args.retro_model_name} "
        f"max_depth={args.max_depth} "
        f"guidance={args.guidance}"
    )

    summary = main(targets, logger, args)
    logger.info("========== Final summary ==========")
    logger.info(
        f"Solved: {summary['success_count']}/{summary['target_count']}; "
        f"errors={summary['error_count']}; missing={summary['missing_count']}"
    )
    logger.info(f"Run summary written: {args.pathway_output_folder}/run_summary.json")
    return int(bool(summary["error_count"] or summary["missing_count"]))


if __name__ == "__main__":
    raise SystemExit(run_cli())
