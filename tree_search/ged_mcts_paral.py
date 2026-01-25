"""ged_mcts_paral module.
"""

import os
import logging
from datetime import datetime
from rdkit import Chem
import argparse
import sys
import time
import csv
import json
import concurrent.futures
import traceback
from tqdm import tqdm

# Import from local mcts module
from .mcts.mcts_ged_controller import MCTS_ged, map_all_reactions, extract_unique_smiles
from .ged_options import ExpandOneOptions, BuildTreeOptions, EnumeratePathsOptions, RetroBackendOption

def setup_logging(log_file):
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

def evaluate_search_success(target_smiles, model_name_list, ged_weight, pathway_output_folder):

    try:
        controller = MCTS_ged(metric_name=str(args.metric_name),
                              ged_weight=float(ged_weight),
                              ged_weight_start=args.ged_weight_start,
                              ged_weight_end=args.ged_weight_end,
                              ged_change_type=args.ged_change_type,
                              track_ged_change=args.track_ged_change,
                              ged_output_folder=args.ged_output_folder,
                              ged_folder_prefix=f"{args.model_name}_{args.ged_change_type}_max_iterations_{args.max_iterations}_strategy_cut_{args.strategy_cut}_strategy_lib_{args.use_strategy_lib}",
                              max_ged=float(args.max_ged),
                              use_strategy_lib=args.use_strategy_lib,
                              strategy_cut=int(args.strategy_cut),
                              deterministic_strategies_path=args.deterministic_strategies_path,
                              ged_weight_in_metric=float(args.ged_weight_in_metric),
                              tanimoto_weight_in_metric=float(args.tanimoto_weight_in_metric),
                              ged_change_type_in_metric=str(args.ged_change_type_in_metric),
                              use_external_temps=args.use_external_temps,
                              external_templates_path=args.external_templates_path,
                              target_smiles=target_smiles)
        # controller = MCTS()
        build_tree_options = BuildTreeOptions()
        build_tree_options.max_iterations = int(args.max_iterations)
        build_tree_options.return_first = True if int(args.return_first) else False
        build_tree_options.expansion_time = int(args.expansion_time)
        build_tree_options.max_depth = int(args.max_depth)
        expand_one_option = ExpandOneOptions()
        expand_one_option.retro_backend_options = []
        for model_name in model_name_list:
            retro_backend_option = RetroBackendOption()
            retro_backend_option.retro_model_name = model_name
            if "strategy" in model_name:
                retro_backend_option.max_num_templates = int(args.strategy_max_num_templates)
            else:
                retro_backend_option.max_num_templates = int(args.one_step_max_num_templates)
            expand_one_option.retro_backend_options.append(retro_backend_option)

        paths, stats, graph = controller.get_buyable_paths(target=target_smiles,
                                                            expand_one_options=expand_one_option,
                                                            build_tree_options=build_tree_options,
                                                            enumerate_paths_options=EnumeratePathsOptions())

        prefix = target_smiles.replace('/', '_').replace("\\", "__")
        output_path = os.path.join(pathway_output_folder, prefix)
        os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, "paths.json"), "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=4, ensure_ascii=False)
        with open(os.path.join(output_path, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        with open(os.path.join(output_path, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=4, ensure_ascii=False)
        if paths:
            reactions = list(extract_unique_smiles(paths[0]))
            reactions = [smarts for smarts in reactions if ">>" in smarts]
            reactions = map_all_reactions(reactions)
            with open(os.path.join(output_path, "unique_smiles.json"), "w", encoding="utf-8") as f:
                json.dump(reactions, f, indent=4, ensure_ascii=False)
        else:
            reactions = controller.latest_rxn_path
            reactions = map_all_reactions(reactions)
            with open(os.path.join(output_path, "failed_smiles.json"), "w", encoding="utf-8") as f:
                json.dump(reactions, f, indent=4, ensure_ascii=False)

        return target_smiles, stats["total_paths"] > 0

    # except Exception as e:
    #     print(f"[Error] Processing {target_smiles} failed: {e}")
    #     return target_smiles, False

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"[Error] Processing {target_smiles} failed: {e}")
        print(f"[Error Details] {error_details}")
        with open(args.multiprocess_bug_smiles_path, "a") as f:
            f.write(f"{target_smiles}\t{str(e)}\t{error_details}\n")
        return target_smiles, False

# [Error] Processing CC(C)c1ccc2c(c1)OC1(O)c3ccccc3C(=O)C21NC(=O)C(=O)c1cccs1 failed: unsupported operand type(s) for +=: 'NoneType' and 'list'

def main(targets, logger, args):
    start_time = time.time()
    logger.info(f"Starting process for {len(targets)} molecule files...")

    success_count = 0
    problematic_targets = ["OC[C@H]1O[C@](O)(c2ccc(Cl)c(Cc3ccc(C#Cc4cnccn4)cc3)c2)[C@H](O)[C@@H](O)[C@@H]1O"]
    targets = [target for target in targets if target not in problematic_targets]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(evaluate_search_success, target, args.model_name_list,
                                   args.ged_weight, args.pathway_output_folder): target for target in targets}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing molecules"):
            target_smiles, success = future.result()
            if success:
                success_count += 1
                logger.info(f"Pathway found for {target_smiles}")
            else:
                logger.info(f"No pathway found for {target_smiles}")

    # if os.path.exists(args.multiprocess_bug_smiles_path):
    #     logger.info("Retrying failed SMILES using single-threaded execution...")
    #     with open(args.multiprocess_bug_smiles_path, "r") as f:
    #         failed_smiles = [line.strip().split("\t")[0] for line in f if line.strip()]
    #     for smi in tqdm(failed_smiles, desc="Retrying failed SMILES"):
    #         _, success = evaluate_search_success(smi, args.model_name_list,
    #                                                 args.ged_weight, args.pathway_output_folder)
    #         if success:
    #             success_count += 1
    #             logger.info(f"[Retry Success] Pathway found for {smi}")
    #         else:
    #             logger.info(f"[Retry Failed] No pathway found for {smi}")

        # Retry failed molecules with iterative multithreaded processing

    if os.path.exists(args.multiprocess_bug_smiles_path):
        logger.info("Retrying failed SMILES using iterative multithreading...")

        max_retry_rounds = 5
        round_num = 1

        while os.path.exists(args.multiprocess_bug_smiles_path) and round_num <= max_retry_rounds:
            with open(args.multiprocess_bug_smiles_path, "r") as f:
                failed_smiles = [line.strip().split("\t")[0] for line in f if line.strip()]
            if not failed_smiles:
                logger.info(f"No more failed SMILES to retry at round {round_num}. Exiting retry loop.")
                break

            open(args.multiprocess_bug_smiles_path, "w").close()
            logger.info(f"[Retry Round {round_num}] Retrying {len(failed_smiles)} molecules...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(evaluate_search_success, smi, args.model_name_list, args.ged_weight, args.pathway_output_folder): smi
                    for smi in failed_smiles
                }
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Retrying Round {round_num}"):
                    try:
                        smi, success = future.result()
                        if success:
                            success_count += 1
                            logger.info(f"[Retry Success] Pathway found for {smi}")
                        else:
                            logger.info(f"[Retry Failed] No pathway found for {smi}")
                    except Exception as e:
                        error_details = traceback.format_exc()
                        logger.warning(f"[Retry Error] Exception during retry for {futures[future]}: {e}")
                        logger.warning(f"[Retry Error Details] {error_details}")

            round_num += 1

    success_rate = (success_count / (len(targets) + len(problematic_targets))) * 100 if len(targets) > 0 else 0
    logger.info(f"Processed {len(targets)} compounds.")
    logger.info(f"Found valid pathways in {success_count} compounds.")
    logger.info(f"Search success rate: {success_rate:.2f}%")
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

def load_mol_files(directory):
    mol_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".mol"):
            mol_path = os.path.join(directory, filename)
            mol_files.append(mol_path)
    return mol_files

def mol_to_smiles(mol_path):
    mol = Chem.MolFromMolFile(mol_path)
    if mol:
        return Chem.MolToSmiles(mol)
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecule datasets and evaluate search success.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    # parser.add_argument("--model_name", type=str, default=None, help="Name of model for expanding nodes")
    parser.add_argument("--model_name_list", type=str, nargs="+", default=None, help="List of model names for expanding nodes")
    parser.add_argument("--ged", action="store_true", help="Whether to use mcts_ged")
    parser.add_argument("--log_output", type=str, default="log_output", help="Log folder")
    parser.add_argument("--ged_weight", type=float, default=0, help="The weight of GED in UCB")
    parser.add_argument("--pathway_output_folder", type=str, default="pathway_output_folder", help="Folder to store pathway output")
    parser.add_argument("--ged_output_folder", type=str, default="ged_output_folder", help="Folder to store ged plots")
    parser.add_argument("--multiprocess_bug_smiles_folder", type=str, default="multiprocess_bug_smiles_folder", help="Folder to store bug smiles in multiprocessing")

    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes to use")
    parser.add_argument("--ged_weight_start", type=float, default=0.05)
    parser.add_argument("--ged_weight_end", type=float, default=0.05)
    parser.add_argument("--ged_change_type", type=str, default="constant")
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--return_first", type=int, default=1, help="whether to stop the search once solving a compound")
    parser.add_argument("--expansion_time", type=int, default=12000)
    parser.add_argument("--max_depth", type=int, default=10)

    parser.add_argument("--max_ged", type=float, default=999)
    parser.add_argument("--strategy_max_num_templates", type=int, default=25)
    parser.add_argument("--one_step_max_num_templates", type=int, default=25)
    parser.add_argument("--strategy_cut", type=int, default=2)
    parser.add_argument("--use_strategy_lib", type=int, default=1, help="whether to use the strategy template library")
    parser.add_argument("--track_ged_change", type=int, default=0, help="whether to track the ged change during iterations")
    parser.add_argument("--deterministic_strategies_type", type=str, default=None)
    parser.add_argument("--metric_name", type=str, default="ged")
    parser.add_argument("--ged_weight_in_metric", type=float, default=0)
    parser.add_argument("--ged_change_type_in_metric", type=str, default="constant")
    parser.add_argument("--tanimoto_weight_in_metric", type=float, default=0)
    parser.add_argument("--external_templates_path", type=str, default=None)
    parser.add_argument("--use_external_temps", type=int, default=0, help="whether to use external templates")
    args = parser.parse_args()

    os.makedirs(args.log_output, exist_ok=True)
    os.makedirs(args.pathway_output_folder, exist_ok=True)
    os.makedirs(args.multiprocess_bug_smiles_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # ged_flag = "ged" if args.ged else ""
    args.model_name = "_".join(args.model_name_list)

    file_name=f"deterministic_strategies_consol_{args.deterministic_strategies_type}.pickle"
    args.deterministic_strategies_path=f"/media/cbe/Data/tanzc/ASKCOSv2/tree_search/mcts/{file_name}"

    log_name = f"{args.dataset_name}_{args.model_name}_metric_{args.metric_name}_hy_ged_{args.ged_weight_in_metric}_hy_tani_{args.tanimoto_weight_in_metric}_{args.ged_change_type}_{args.ged_weight}_max_iterations_{args.max_iterations}_max_depth_{args.max_depth}_strategy_cut_{args.strategy_cut}_strategy_lib_{args.use_strategy_lib}_{args.deterministic_strategies_type}_{timestamp}.log"
    log_file = os.path.join(args.log_output, log_name)
    logger = setup_logging(log_file)

    args.multiprocess_bug_smiles_path = os.path.join(args.multiprocess_bug_smiles_folder, log_name.replace(".log", ".txt"))

    targets = []
    if args.dataset_name == "demo_targets":
        demo_file_path = os.path.join("data", "demo_targets.csv")
        if not os.path.exists(demo_file_path):
            raise FileNotFoundError(f"Demo dataset not found: {demo_file_path}. Please run scripts/demo.py first to create it.")
        with open(demo_file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0].strip():  # Skip empty lines
                    targets.append(row[0].strip())
    elif args.dataset_name == "uspto_190":
        with open("uspto_190_targets.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                targets.append(row[0])

    ReTReK_dataset_list = ["chembl", "drug-like-compounds", "chematica"]
    if args.dataset_name in ReTReK_dataset_list:
        ReTReK_dir = f"ReTReK/data/evaluation_compounds/{args.dataset_name}"
        mol_files = load_mol_files(ReTReK_dir)
        for mol_file in mol_files:
            smiles = mol_to_smiles(mol_file)
            if smiles:
                targets.append(smiles)
    elif args.dataset_name == "chembl_35_random_smiles_100000":
        file_path = '/media/cbe/Data/tanzc/ASKCOSv2/tree_search/ged_repo/chembl_35_random_smiles_100000.txt'
        with open(file_path, 'r') as file:
            for line in file:
                smile = line.strip()
                if smile:
                    targets.append(smile)

    elif args.dataset_name == "coconut":
        file_path = "/home/cbe/workspace/tanzc/key_step_guide/coconut_smiles_only.txt"
        with open(file_path, 'r') as file:
            for line in file:
                smile = line.strip()
                if smile:
                    targets.append(smile)

    elif args.dataset_name == "reaxys_1000":
        file_path = '/media/cbe/Data/tanzc/ASKCOSv2/tree_search/ged_repo/new_cleaned_test_set_reaxys.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        targets = data
    elif args.dataset_name == "drugs_and_natural_products":
        import pandas as pd
        file_path = "/media/cbe/Data/tanzc/ASKCOSv2/tree_search/ged_repo/drugs_and_natural_products_smiles.csv"
        df = pd.read_csv(file_path)
        data = df["SMILES"].to_list()
        targets = data
    elif args.dataset_name == "drugs_and_natural_products_Roh":
        import pandas as pd
        file_path = "/media/cbe/Data/tanzc/ASKCOSv2/tree_search/ged_repo/drugs_and_natural_products_smiles_Roh.csv"
        df = pd.read_csv(file_path)
        data = df["SMILES"].to_list()
        targets = data
    elif args.dataset_name == "intermediates_Mannich_pathway_Lin":
       with open("intermediates_Mannich_pathway_Lin.txt", 'r') as file:
            for line in file:
                smile = line.strip()
                if smile:
                    targets.append(smile)

    elif args.dataset_name == "error_only":
        file_path = "/media/cbe/Data/tanzc/ASKCOSv2/tree_search/ged_repo/multiprocess_bug_smiles_folder/uspto_190_uspto_original_consol_Roh_metric_SA_Score&SC_Score&ged_hy_ged_0.005_hy_tani_0_logarithmic_change_0.8_max_iterations_500_max_depth_10_strategy_cut_1_strategy_lib_0_uspto_full_2025-08-09_13-45-28.txt"
        with open(file_path, 'r') as file:
            failed_smiles = [line.strip().split("\t")[0] for line in file if line.strip()]
        targets = failed_smiles

    elif args.dataset_name == "unsolved_smiles_uspto_190_53":
        with open("/home/cbe/workspace/tanzc/ASKCOSv2/tree_search/transform_goal/unsolved_smiles_uspto_190_53.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                targets.append(row[0])

    main(targets, logger, args)

