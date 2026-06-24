"""Parallel uspto-style runner using official binghong-ml/retro_star RSPlanner."""
import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

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
    load_targets as load_experiment_targets,
    resolve_targets_path,
    write_run_summary,
)
from tree_search.retro_star.adapters.depth_utils import (
    binghong_route_max_mol_depth,
    route_depth_within_limit,
)

_WORKER_PLANNER = None
_SHARED_STARTING_MOLECULES = None
_SHARED_STARTING_MOLECULES_PATH = None


def _import_rs_planner():
    import sys

    argv = sys.argv
    sys.argv = ["binghong_retro_star"]
    try:
        from retro_star.api import RSPlanner
        return RSPlanner
    finally:
        sys.argv = argv


def _import_prepare_starting_molecules():
    import sys

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


def _resolve_search_max_depth(args):
    if getattr(args, "search_max_depth", None) is not None:
        val = int(args.search_max_depth)
        return val if val >= 0 else None
    return None


def _resolve_enumerate_max_depth(args):
    if getattr(args, "enumerate_max_depth", None) is not None:
        return int(args.enumerate_max_depth)
    if getattr(args, "max_depth", None) is not None:
        return int(args.max_depth)
    return 10


def _init_worker(
    gpu,
    use_value_fn,
    iterations,
    expansion_topk,
    search_max_depth,
    starting_molecules,
    one_step_backend,
    retro_model_name,
    metric_name,
    metric_weight_start,
    metric_weight_end,
    metric_schedule_type,
):
    global _WORKER_PLANNER
    RSPlanner = _import_rs_planner()
    _patch_rsplanner_starting_molecules_loader(starting_molecules)
    # binghong's parse_args (run at import, with scrubbed argv) force-sets
    # CUDA_VISIBLE_DEVICES from its OWN default (-1), hiding all GPUs. Restore the
    # requested GPU here -- in the worker, post-fork, before any CUDA context is created --
    # so the value network and the one-step ("auto") model land on the GPU. Each worker
    # builds its own CUDA context after fork, which is the fork-safe pattern.
    if int(gpu) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    _WORKER_PLANNER = RSPlanner(
        gpu=int(gpu),
        use_value_fn=bool(int(use_value_fn)),
        iterations=int(iterations),
        expansion_topk=int(expansion_topk),
        search_max_depth=search_max_depth,
        starting_molecules=starting_molecules,
        one_step_backend=one_step_backend,
        retro_model_name=retro_model_name,
        metric_name=(metric_name or None),
        metric_weight_start=float(metric_weight_start),
        metric_weight_end=float(metric_weight_end),
        metric_schedule_type=str(metric_schedule_type),
    )


def evaluate_search_success(target_smiles, args):
    """Plan one target with the worker's RSPlanner and write its route/stats outputs.

    Returns ``(target_smiles, outcome)`` where ``outcome`` carries search/enumerate success flags.
    """
    import gc
    import traceback

    gc.collect()
    bounded = _resolve_search_max_depth(args) is not None
    enumerate_max_depth = _resolve_enumerate_max_depth(args)

    try:
        result = _WORKER_PLANNER.plan(target_smiles)
        search_succ = bool(result and result.get("succ"))
        route_len = int(result.get("route_len", 0)) if result else None
        route_str = result.get("routes") if result else None
        route_max_mol_depth = (
            binghong_route_max_mol_depth(route_str) if route_str else None
        )

        if bounded:
            enumerate_succ = search_succ
        else:
            enumerate_succ = search_succ and route_depth_within_limit(
                route_max_mol_depth, enumerate_max_depth
            )

        prefix = target_smiles.replace("/", "_").replace("\\", "__")
        output_path = os.path.join(args.pathway_output_folder, prefix)
        os.makedirs(output_path, exist_ok=True)

        stats = {
            "succ": enumerate_succ,
            "search_succ": search_succ,
            "enumerate_succ": enumerate_succ,
            "iterations": int(result.get("iter", 0)) if result else 0,
            "time": float(result.get("time", 0.0)) if result else 0.0,
            "route_cost": result.get("route_cost") if result else None,
            "route_len": route_len,
            "route_max_mol_depth": route_max_mol_depth,
            "search_max_depth": _resolve_search_max_depth(args),
            "enumerate_max_depth": None if bounded else enumerate_max_depth,
            "depth_within_limit": enumerate_succ,
        }
        paths = []
        if enumerate_succ and result and result.get("routes") is not None:
            paths = [{"binghong_route": result["routes"]}]

        with open(os.path.join(output_path, "paths.json"), "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=4, ensure_ascii=False)
        with open(os.path.join(output_path, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        gc.collect()
        return target_smiles, {
            "search_succ": search_succ,
            "enumerate_succ": enumerate_succ,
            "stats": stats,
        }

    except Exception as e:
        print(f"[Error] Processing {target_smiles} failed: {e}")
        traceback.print_exc()
        return target_smiles, {"search_succ": False, "enumerate_succ": False, "stats": {}}


def main(targets, logger, args):
    """Plan every target across a worker pool and return the run summary dict."""
    start = time.time()
    bounded = _resolve_search_max_depth(args) is not None
    search_success_count = 0
    enumerate_success_count = 0
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
        _resolve_search_max_depth(args),
        starting_molecules,
        str(args.one_step_backend),
        str(args.retro_model_name),
        (str(args.metric_name) if args.metric_name else None),
        float(args.metric_weight_start),
        float(args.metric_weight_end),
        str(args.metric_schedule_type),
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=initargs,
    ) as ex:
        futures = {ex.submit(evaluate_search_success, t, args): t for t in targets}
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="binghong Retro*",
        ):
            try:
                smi, outcome = fut.result(timeout=7200)
                stats = outcome.get("stats", {})
                if outcome["search_succ"]:
                    search_success_count += 1
                if outcome["enumerate_succ"]:
                    enumerate_success_count += 1
                if outcome["enumerate_succ"]:
                    logger.info(
                        f"Pathway found for {smi} "
                        f"(iter={stats.get('iterations', '?')}, "
                        f"time={stats.get('time', 0):.2f}s)"
                    )
                elif outcome["search_succ"] and not bounded:
                    logger.info(
                        f"Pathway found for {smi} (exceeds enumerate depth limit, "
                        f"iter={stats.get('iterations', '?')})"
                    )
                else:
                    logger.info(f"No pathway found for {smi}")
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    n = len(targets)
    search_rate = (search_success_count / n * 100) if n else 0.0
    enumerate_rate = (enumerate_success_count / n * 100) if n else 0.0
    elapsed_seconds = time.time() - start
    logger.info(f"Processed {n} compounds.")

    if bounded:
        logger.info(
            f"Success (search max_depth={_resolve_search_max_depth(args)}): "
            f"{search_success_count}/{n} ({search_rate:.2f}%)"
        )
        primary_count = search_success_count
        primary_rate = search_rate
    else:
        logger.info(
            f"Search success (no depth limit): {search_success_count}/{n} ({search_rate:.2f}%)"
        )
        logger.info(
            f"Enumerate success (depth limit): {enumerate_success_count}/{n} ({enumerate_rate:.2f}%)"
        )
        primary_count = enumerate_success_count
        primary_rate = enumerate_rate

    logger.info(f"Total time: {elapsed_seconds:.2f}s")
    return {
        "target_count": n,
        "search_success_count": search_success_count,
        "search_success_rate": search_rate,
        "enumerate_success_count": enumerate_success_count,
        "enumerate_success_rate": enumerate_rate,
        "success_count": primary_count,
        "failed_count": n - primary_count,
        "success_rate": primary_rate,
        "elapsed_seconds": elapsed_seconds,
        "depth_mode": "bounded_search" if bounded else "unbounded_search",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ``run_retro_star`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Parallel binghong official Retro* (RSPlanner) search."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--log_output", type=str, default="log_output")
    parser.add_argument("--pathway_output_folder", type=str, default="binghong_retro_star_outputs")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=500)
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
        "--search_max_depth",
        type=int,
        default=None,
        help="If set (e.g. 10), cap depth during search (bounded mode).",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Post-filter depth for unbounded mode statistics (enumerate_max_depth).",
    )
    parser.add_argument("--enumerate_max_depth", type=int, default=None)
    parser.add_argument("--use_value_network", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--targets_file", type=str, default=None)
    parser.add_argument(
        "--metric_name",
        type=str,
        default=None,
        help="Optional metric guidance, e.g. 'sa_score&sc_score&ged' (template_relevance backend only).",
    )
    parser.add_argument("--metric_weight_start", type=float, default=0.0)
    parser.add_argument("--metric_weight_end", type=float, default=0.0)
    parser.add_argument(
        "--metric_schedule_type",
        type=str,
        default="constant",
        help="Iteration schedule for the metric weight (constant, linear, exponential, logarithmic, sigmoid).",
    )
    return parser


def run_cli(argv=None) -> None:
    """Parse args, run the batch Retro* search, and write ``run_summary.json``. CLI entry point."""
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

    bounded = _resolve_search_max_depth(args) is not None
    depth_tag = (
        f"searchd{args.search_max_depth}"
        if bounded
        else f"enumd{_resolve_enumerate_max_depth(args)}"
    )
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(
        args.log_output,
        f"binghong_retro_star_{args.dataset_name}_vn{args.use_value_network}_"
        f"topk{args.expansion_topk}_iter{args.max_iterations}_{depth_tag}_{ts}.log",
    )
    logger = setup_logging(log_file)

    targets_path = resolve_targets_path(args.dataset_name, args.targets_file)
    targets = load_experiment_targets(args.dataset_name, args.targets_file)
    logger.info(
        f"Loaded {len(targets)} targets from {targets_path}; "
        f"binghong RSPlanner vn={args.use_value_network} "
        f"topk={args.expansion_topk} iter={args.max_iterations} "
        f"buyables={args.buyables_path} "
        f"one_step_backend={args.one_step_backend} "
        f"retro_model_name={args.retro_model_name} "
        f"search_max_depth={_resolve_search_max_depth(args)}"
    )

    summary = main(targets, logger, args)
    summary.update(
        {
            "backend": "binghong_retro_star.RSPlanner",
            "dataset_name": args.dataset_name,
            "targets_file": str(targets_path),
            "num_workers": int(args.num_workers),
            "max_iterations": int(args.max_iterations),
            "expansion_topk": int(args.expansion_topk),
            "buyables_path": os.path.abspath(args.buyables_path),
            "one_step_backend": str(args.one_step_backend),
            "retro_model_name": str(args.retro_model_name),
            "metric_name": (str(args.metric_name) if args.metric_name else None),
            "metric_weight_start": float(args.metric_weight_start),
            "metric_weight_end": float(args.metric_weight_end),
            "metric_schedule_type": str(args.metric_schedule_type),
            "use_value_network": bool(int(args.use_value_network)),
            "gpu": int(args.gpu),
            "search_max_depth": _resolve_search_max_depth(args),
            "enumerate_max_depth": (
                None if bounded else _resolve_enumerate_max_depth(args)
            ),
            "depth_applies_during": "search" if bounded else "enumerate_only",
        }
    )
    write_run_summary(args.pathway_output_folder, summary)
    logger.info("========== Final summary ==========")
    if bounded:
        logger.info(
            f"Success rate: {summary['success_rate']:.2f}% "
            f"({summary['success_count']}/{summary['target_count']})"
        )
    else:
        enum_depth = summary.get("enumerate_max_depth")
        depth_note = f" (max mol depth < {enum_depth})" if enum_depth is not None else ""
        logger.info(
            f"Search success rate: {summary['search_success_rate']:.2f}% "
            f"({summary['search_success_count']}/{summary['target_count']})"
        )
        logger.info(
            f"Enumerate success rate{depth_note}: {summary['enumerate_success_rate']:.2f}% "
            f"({summary['enumerate_success_count']}/{summary['target_count']})"
        )
    logger.info(f"Run summary written: {args.pathway_output_folder}/run_summary.json")


if __name__ == "__main__":
    run_cli()
