"""ged_mcts_paral module - Parallel processing for GED-guided MCTS retrosynthesis.
"""
import os
import sys

# Ensure project root is in path FIRST - before any other imports
# This is critical for multiprocessing subprocesses
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
from datetime import datetime
from rdkit import Chem
import argparse
import time
import csv
import json
import concurrent.futures
from tqdm import tqdm


def setup_logging(log_file):
    """Setup logging configuration."""
    logger = logging.getLogger("MolProcessLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def evaluate_search_success(target_smiles, model_name_list, args):
    """Evaluate search success for a single target SMILES."""
    import sys
    import os
    import traceback
    import gc
    
    # Force garbage collection at start to free up memory
    gc.collect()
    
    # Ensure project root is in path for multiprocessing subprocesses
    # Use current working directory (should be project root when script is run)
    _project_root = os.getcwd()
    # Verify it's the right directory by checking for tree_search
    if not os.path.exists(os.path.join(_project_root, 'tree_search', 'mcts')):
        # Try parent directories
        parent = os.path.dirname(_project_root)
        if os.path.exists(os.path.join(parent, 'tree_search', 'mcts')):
            _project_root = parent
        else:
            # Last resort: walk up to find tree_search
            current = _project_root
            for _ in range(5):  # Max 5 levels up
                if os.path.exists(os.path.join(current, 'tree_search', 'mcts')):
                    _project_root = current
                    break
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
    
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    
    from tree_search.mcts.mcts_ged_controller import MCTS_ged
    from tree_search.mcts.options import ExpandOneOptions, BuildTreeOptions, EnumeratePathsOptions, RetroBackendOption
    
    try:
            controller = MCTS_ged(
            metric_name=str(args.metric_name),
            ged_weight=float(args.ged_weight),
            ged_weight_start=args.ged_weight_start,
            ged_weight_end=args.ged_weight_end,
            ged_change_type=args.ged_change_type,
            ged_weight_in_metric=float(args.ged_weight_in_metric),
            tanimoto_weight_in_metric=float(args.tanimoto_weight_in_metric),
            ged_change_type_in_metric=str(args.ged_change_type_in_metric)
            )
        
            build_tree_options = BuildTreeOptions()
            build_tree_options.max_iterations = int(args.max_iterations)
            build_tree_options.return_first = bool(int(args.return_first))
            build_tree_options.expansion_time = int(args.expansion_time)
            build_tree_options.max_depth = int(args.max_depth)
        
            expand_one_option = ExpandOneOptions()
            expand_one_option.retro_backend_options = []
            for model_name in model_name_list:
                retro_backend_option = RetroBackendOption()
                retro_backend_option.retro_model_name = model_name
                retro_backend_option.max_num_templates = int(args.one_step_max_num_templates)
                expand_one_option.retro_backend_options.append(retro_backend_option)

            paths, stats, graph = controller.get_buyable_paths(
                target=target_smiles,
                expand_one_options=expand_one_option,
                build_tree_options=build_tree_options,
                enumerate_paths_options=EnumeratePathsOptions()
            )

            prefix = target_smiles.replace('/', '_').replace("\\", "__")
            output_path = os.path.join(args.pathway_output_folder, prefix)
            os.makedirs(output_path, exist_ok=True)

            with open(os.path.join(output_path, "paths.json"), "w", encoding="utf-8") as f:
                json.dump(paths, f, indent=4, ensure_ascii=False)
            with open(os.path.join(output_path, "stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            with open(os.path.join(output_path, "graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=4, ensure_ascii=False)

            # Clean up before returning
            gc.collect()
            return target_smiles, stats["total_paths"] > 0

    except Exception as e:
        error_msg = f"[Error] Processing {target_smiles} failed: {e}"
        print(error_msg)
        # Print full traceback for debugging
        traceback.print_exc()
        return target_smiles, False


def main(targets, logger, args):
    """Main processing function."""
    start_time = time.time()
    logger.info(f"Starting process for {len(targets)} molecules...")

    success_count = 0
    problematic_targets = ["OC[C@H]1O[C@](O)(c2ccc(Cl)c(Cc3ccc(C#Cc4cnccn4)cc3)c2)[C@H](O)[C@@H](O)[C@@H]1O"]
    targets = [target for target in targets if target not in problematic_targets]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(evaluate_search_success, target, args.model_name_list, args): target
            for target in targets
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing molecules"):
            try:
                target_smiles, success = future.result(timeout=3600)  # 1 hour timeout per molecule
                if success:
                    success_count += 1
                    logger.info(f"Pathway found for {target_smiles}")
                else:
                    logger.info(f"No pathway found for {target_smiles}")
            except concurrent.futures.TimeoutError:
                target_smiles = futures.get(future, 'unknown')
                logger.error(f"Timeout processing {target_smiles}")
                logger.info(f"No pathway found for {target_smiles} (timeout)")
            except concurrent.futures.process.BrokenProcessPool:
                target_smiles = futures.get(future, 'unknown')
                logger.error(f"Process crashed while processing {target_smiles} - likely due to memory issue or unhandled exception")
                logger.info(f"No pathway found for {target_smiles} (process crashed)")
            except Exception as e:
                import traceback
                target_smiles = futures.get(future, 'unknown')
                logger.error(f"Error getting result for {target_smiles}: {e}")
                traceback.print_exc()
                logger.info(f"No pathway found for {target_smiles} (error)")

    success_rate = (success_count / (len(targets) + len(problematic_targets))) * 100 if len(targets) > 0 else 0
    logger.info(f"Processed {len(targets)} compounds.")
    logger.info(f"Found valid pathways in {success_count} compounds.")
    logger.info(f"Search success rate: {success_rate:.2f}%")
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")


def load_targets(dataset_name):
    """Load target molecules from dataset."""
    targets = []
    
    if dataset_name == "demo_targets":
        demo_file_path = os.path.join("data", "demo_targets.csv")
        if not os.path.exists(demo_file_path):
            raise FileNotFoundError(f"Demo dataset not found: {demo_file_path}")
        with open(demo_file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0].strip():
                    targets.append(row[0].strip())
                    
    elif dataset_name == "uspto_190":
        uspto_file_path = os.path.join("data", "uspto_190_targets.csv")
        if not os.path.exists(uspto_file_path):
            # Fallback to current directory
            uspto_file_path = "uspto_190_targets.csv"
        if os.path.exists(uspto_file_path):
            with open(uspto_file_path, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0].strip():
                        targets.append(row[0].strip())
        else:
            raise FileNotFoundError(f"USPTO 190 dataset not found: {uspto_file_path}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: demo_targets, uspto_190")
    
    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecule datasets and evaluate search success.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process (demo_targets, uspto_190)")
    parser.add_argument("--model_name_list", type=str, nargs="+", required=True, help="List of model names for expanding nodes")
    parser.add_argument("--log_output", type=str, default="log_output", help="Log folder")
    parser.add_argument("--ged_weight", type=float, default=0, help="The weight of GED in UCB")
    parser.add_argument("--pathway_output_folder", type=str, default="pathway_output_folder", help="Folder to store pathway output")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes to use")
    parser.add_argument("--ged_weight_start", type=float, default=0.05)
    parser.add_argument("--ged_weight_end", type=float, default=0.05)
    parser.add_argument("--ged_change_type", type=str, default="constant")
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--return_first", type=int, default=1, help="whether to stop the search once solving a compound")
    parser.add_argument("--expansion_time", type=int, default=12000)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--one_step_max_num_templates", type=int, default=25)
    parser.add_argument("--metric_name", type=str, default="ged")
    parser.add_argument("--ged_weight_in_metric", type=float, default=0)
    parser.add_argument("--ged_change_type_in_metric", type=str, default="constant")
    parser.add_argument("--tanimoto_weight_in_metric", type=float, default=0)
    
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.log_output, exist_ok=True)
    os.makedirs(args.pathway_output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_name = "_".join(args.model_name_list)

    log_name = f"{args.dataset_name}_{args.model_name}_metric_{args.metric_name}_hy_ged_{args.ged_weight_in_metric}_hy_tani_{args.tanimoto_weight_in_metric}_{args.ged_change_type}_{args.ged_weight}_max_iterations_{args.max_iterations}_max_depth_{args.max_depth}_{timestamp}.log"
    log_file = os.path.join(args.log_output, log_name)
    logger = setup_logging(log_file)

    targets = load_targets(args.dataset_name)
    logger.info(f"Loaded {len(targets)} target molecules from {args.dataset_name}")

    main(targets, logger, args)
