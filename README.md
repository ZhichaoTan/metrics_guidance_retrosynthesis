# Guiding Computer-Aided Multistep Retrosynthesis with Complexity and Distance Logic

This repository contains the implementation for Guiding Computer-Aided Multistep Retrosynthesis with Complexity and Distance Logic.

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Code Structure](#code-structure)
- [Guiding Metric Calculation](#guiding-metric-calculation)
- [Metric-guided Retrosynthetic Search](#metric-guided-retrosynthetic-search)
- [Acknowledgments](#acknowledgments)

## Overview

This implementation consists of two core modules:

1. **Guiding metric calculation**: Automates the calculation of graph edit distance (GED) between intermediates and the target molecule, following the manual approach described in the [Science paper](https://www.science.org/doi/10.1126/science.ade8459). GED calculation supports both reaction-level ([`ged_scorer.py`](metrics/ged_scorer.py)) and pathway-level ([`syn_tree_ged.py`](metrics/syn_tree_ged.py)) measurements. This implementation also supports complexity metrics (sa_score and sc_score) and a distance metric (Tanimoto distance).

2. **Metric-guided retrosynthetic search**: Integrates the metric calculations into the search to guide retrosynthetic exploration, using the chemistry guidance together with the one-step model. The one-step model is the **USPTO full consolidated** template-relevance model from [higherlev_retro](https://github.com/jihye-roh/higherlev_retro). **This implementation supports local deployment without requiring Docker or external servers.** We adapt corresponding modules from [ASKCOS](https://gitlab.com/mlpds_mit/askcosv2), and these modules can also be integrated into ASKCOS.

## System Requirements

The codebase has been developed and tested on Linux. The machine used for development and the reported runs:

- **Operating System:** Ubuntu 24.04 LTS
- **CPU:** AMD Ryzen 9 9950X (16 cores / 32 threads)
- **RAM:** 64 GB
- **GPU:** NVIDIA GeForce RTX 5090 (24 GB each), CUDA 12.9 — optional, used only to accelerate one-step model inference
- **Python:** 3.10.13

A GPU is optional; the search runs on CPU. For parallel multistep retrosynthesis, a multi-core CPU (≥10 cores) and ≥16 GB RAM are recommended.

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/ZhichaoTan/metrics_guidance_retrosynthesis
cd metrics_guidance_retrosynthesis
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate metrics_guidance_retrosynthesis
```

### Step 3: Download and Extract the One-Step Model

Download the [`uspto_original_consol.mar` file](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ztanaj_connect_ust_hk/IQD3QJWNq_prQppG2VtJLTDRAYRK4WIuLVGXSx9WK9a3Bz4?e=1z5MeS) on OneDrive — the USPTO full consolidated one-step model from the [higherlev_retro repository](https://github.com/jihye-roh/higherlev_retro). The `.mar` file is a zip archive; extract it with:

```bash
unzip uspto_original_consol.mar -d tree_search/uspto_original_consol_Roh
```

This unpacks the model files into `tree_search/uspto_original_consol_Roh/`. After extraction it should contain at least:
- `model_latest.pt` - The trained model weights
- `templates.jsonl` - Template library
- `models.py` - Model architecture definitions
- `utils.py` - Model utilities

> **Note:** The loader passes `weights_only=False` to `torch.load` automatically, which this checkpoint requires on PyTorch ≥ 2.6.

## Code Structure

```
metrics_guidance_retrosynthesis/
├── tree_search/                              # Retrosynthetic search module
│   ├── run_mcts.py                           # Public entry point: metric-guided MCTS
│   ├── run_retro_star.py                     # Public entry point: metric-guided Retro*
│   ├── ged_mcts_paral.py                     # Parallel MCTS batch driver
│   ├── retro_star_binghong_paral.py          # Parallel Retro* (RSPlanner) batch driver
│   ├── experiment_utils.py                   # Dataset loading and batch helpers
│   ├── buyables.json.gz                      # Buyable molecules database from ASKCOS
│   ├── mcts/                                 # MCTS implementation
│   │   ├── mcts_controller.py                # Base MCTS controller
│   │   ├── mcts_ged_controller.py            # Metric-guided MCTS controller (PUCT modes)
│   │   ├── metric_aggregation_utils.py       # Aggregate sibling metric values into a chemistry score
│   │   ├── metric_weight_schedule.py         # Guidance-weight schedule functions
│   │   ├── options.py                        # MCTS configuration options
│   │   ├── utils_mcts.py                     # MCTS utility functions
│   │   └── modules/                          # Local one-step / pricer APIs (ASKCOS-adapted)
│   │       ├── expand_one_api.py
│   │       ├── expand_one_controller.py
│   │       ├── local_expand_one_api.py
│   │       ├── local_pricer_api.py
│   │       ├── local_retro_controller.py
│   │       ├── local_templ_rel_predictor.py
│   │       ├── templ_rel_predictor.py
│   │       └── templ_rel_parser.py
│   ├── retro_star/                           # Metric-guided Retro* (binghong RSPlanner adapter)
│   │   ├── options.py
│   │   ├── adapters/                         # Roh one-step adapter + depth utilities
│   │   └── modules/                          # Local expansion adapter + module loader
│   └── uspto_original_consol_Roh/            # One-step model dir (from .mar; git-ignored)
├── metrics/                                  # Metric calculation modules
│   ├── ged_scorer.py                         # Reaction-level GED calculation
│   ├── syn_tree_ged.py                       # Pathway-level GED / metric trajectory
│   ├── sascore.py                            # sa_score calculation
│   ├── scscore.py                            # sc_score calculation
│   ├── linear_tree_dfs.py                    # Tree traversal utilities
│   ├── askcos_pickle.py                      # ASKCOS data utilities
│   ├── sa_score_model/fpscores.pkl.gz        # sa_score fingerprint scores
│   └── sc_score_model/model_1024bool.pickle  # sc_score model weights
├── datastructs/                             # Data structure definitions
│   ├── datastructs.py                        # Core data structures
│   └── syn_tree.py                           # Synthesis tree data structure
├── data/                                    # Datasets (all replaceable via --targets_file)
│   ├── demo_targets.csv                      # Demo dataset (5 molecules)
│   ├── uspto_190_targets.csv                 # USPTO-190 dataset (190 molecules)
│   ├── drug_hunter_172.txt                   # Drug Hunter dataset (172 molecules)
│   └── drugs_and_natural_products.csv        # Drugs & natural products (21 molecules)
├── scripts/                                 # Reproduction scripts
│   ├── mcts_uspto_190.sh                      # MCTS metric-hybrid PUCT on USPTO-190 (core setting)
│   ├── retro_star_uspto_190.sh               # Metric-guided Retro* on USPTO-190
│   ├── demo_single_process.sh                # Single-process demo smoke test
│   └── patches/binghong_retro_star.patch     # Required source-change record for Retro*
├── utils/
│   └── chem_utils.py                         # Chemistry utilities (SMILES canonicalization, mapping)
├── environment.yml                          # Conda environment configuration
├── test_metric_notebook.ipynb               # Notebook demonstrating metric calculation
└── README.md                                # This file
```

## Guiding Metric Calculation

This module provides automated calculation of the metrics used to guide retrosynthetic search: complexity metrics (sa_score, sc_score) and distance metrics (Tanimoto distance, Graph Edit Distance). A runnable walkthrough is in [`test_metric_notebook.ipynb`](test_metric_notebook.ipynb).

### Existing Guiding Metrics

**sa_score** and **Tanimoto distance** are implemented directly with RDKit. **sc_score** is calculated using the [reference repository](https://github.com/connorcoley/scscore).

### Graph Edit Distance (GED)

We implement GED at two levels: reaction-level and pathway-level. GED measures the number of bond and/or chirality edits required to transform an intermediate into the target molecule.

#### Reaction-level GED

The reaction-level GED quantifies the structural change occurring in a single reaction step. It is implemented in [`ged_scorer.py`](metrics/ged_scorer.py).

**Important**: Atoms between reactants and products must be mapped before input. We use [RXNMapper](https://github.com/rxn4chemistry/rxnmapper) for automatic atom mapping, but any atom-mapper can be substituted — the scorer only requires the input reaction SMILES to carry consistent atom maps.

```python
from metrics.ged_scorer import GEDScorer

# Example reaction SMILES with atom mapping
reaction_smiles = "[C:11](=[O:12])([CH2:13][CH2:14][CH:19]=[O:20])[OH:18].[CH3:1][C@H:2]([CH2:3][OH:4])[CH2:16][CH:6]=[O:5].[NH2:10][CH3:15]>>[CH3:1][C@H:2]([CH2:3][OH:4])[C@H:16]([CH:6]=[O:5])[C@H:15]([NH2:10])[CH2:14][CH2:13][C:11](=[O:12])[OH:18]"

score = GEDScorer(reaction_smiles).calculate_score()
print("GED score for this reaction:", score)   # -> 5.0
```

#### Pathway-level GED

The pathway-level GED evaluates the structural changes along an entire synthetic route. It is implemented in [`syn_tree_ged.py`](metrics/syn_tree_ged.py), which extends the synthesis tree with metric-tracking capabilities. Passing `add_ged_init=True` annotates every reaction edge with its GED and records the per-step trajectory.

```python
from metrics.syn_tree_ged import SynTreeGED

# Example retrosynthetic route.
# If reactions are not atom-mapped, they are mapped automatically with RXNMapper (replaceable).
example_route = [
    "C.C=CCC=O.O>>[H]C(=O)C[C@H](C)CO",
    "COC1=CC=C(C=C1)N.O=CCCC(=O)O.[H]C(=O)C[C@H](C)CO>>[H]C(=O)[C@]([H])([C@H](C)CO)[C@@H](N)CCC(=O)O",
    "C=CC.[H]C(=O)[C@]([H])([C@H](C)CO)[C@@H](N)CCC(=O)O>>[H][C@]([C@H](C)CO)([C@H](O)CC=C)[C@@]([H])(N)CCC(=O)O",
]

syn_tree = SynTreeGED(route=example_route, track_atom_init=True, add_ged_init=True)

# Per-step GED trajectory along the pathway
print(syn_tree.one_step_delta_metric_list)    # -> [4.0, 7.0, 3.0]
print(syn_tree.pairwise_delta_metric_list)     # -> [4.0, 7.0, 3.0]

# Other metrics can be annotated on the same tree
syn_tree.add_pairwise_attributes(metric_name="sa_score")
```

## Metric-guided Retrosynthetic Search

This module integrates the metrics into the search process. All components run locally, with no Docker or external server required. Main interfaces are listed below:

- **Guidance modes** (`--guidance_mode`): `model_prior_puct` (baseline) or `metric_hybrid_puct` (metric-guided).
- **Metric aggregation** (`--metric_aggregation_mode`): `rank_aggregation` or `signed_rank_positive_only`. This controls how the sibling metric values are aggregated into the chemistry score `G`. In MCTS, `G` is blended into the PUCT score as a convex term `(1 − k)·PUCT + k·G`.
- **Weight schedule** (`--metric_schedule_type`): `constant`, `linear`, `exponential`, `logarithmic`, or `sigmoid` — how the guidance weight `k` ramps over iterations.

### Quick Start Demo

The demo runs the core setting on the 5 bundled demo molecules with a single worker and 100 MCTS iterations.

```bash
bash scripts/demo_single_process.sh
```

Equivalently:

```bash
python -m tree_search.run_mcts \
    --dataset_name demo_targets \
    --model_name_list uspto_original_consol_Roh \
    --metric_name "sa_score&sc_score&ged" \
    --guidance_mode metric_hybrid_puct \
    --metric_aggregation_mode signed_rank_positive_only \
    --metric_weight_start 0 \
    --metric_weight_end 1 \
    --metric_schedule_type logarithmic \
    --max_iterations 100 \
    --max_depth 10 \
    --one_step_max_num_templates 25 \
    --num_workers 1 \
    --pathway_output_folder outputs/readme_demo_single_process_100
```

Each target gets a subdirectory under `--pathway_output_folder` containing `paths.json` (found routes), `stats.json` (search statistics), and `graph.json` (search graph); a batch summary is written to `run_summary.json`. Logs go to `log_output/`.

### USPTO-190 Reproduction

The full dataset `data/uspto_190_targets.csv` contains the 190 target molecules used in the study. Reproduce the quantitative results with the maintained script, which uses the core configuration above (10 parallel workers by default):

```bash
bash scripts/mcts_uspto_190.sh
```

Or run manually:

```bash
python -m tree_search.run_mcts \
    --dataset_name uspto_190 \
    --model_name_list uspto_original_consol_Roh \
    --metric_name "sa_score&sc_score&ged" \
    --guidance_mode metric_hybrid_puct \
    --metric_aggregation_mode signed_rank_positive_only \
    --metric_weight_start 0 \
    --metric_weight_end 1 \
    --metric_schedule_type logarithmic \
    --max_iterations 500 \
    --max_depth 10 \
    --one_step_max_num_templates 25 \
    --num_workers 10 \
    --pathway_output_folder outputs/uspto190_core_mcts
```

### Baseline (No Metric)

To run the PUCT baseline, set `--guidance_mode model_prior_puct`. This selects on the one-step model prior alone, so no metric is computed and all metric-related arguments are omitted:

```bash
python -m tree_search.run_mcts \
    --dataset_name demo_targets \
    --model_name_list uspto_original_consol_Roh \
    --guidance_mode model_prior_puct \
    --max_iterations 100 \
    --max_depth 10 \
    --one_step_max_num_templates 25 \
    --num_workers 1 \
    --pathway_output_folder outputs/readme_baseline_single_process_100
```

### Datasets

Four datasets are bundled and selected by `--dataset_name`:

| Name | Size | Description |
|------|------|-------------|
| `demo_targets` | 5 | Small demo subset |
| `uspto_190` | 190 | USPTO-190 benchmark |
| `drug_hunter_172` | 172 | Drug Hunter dataset (from [SimpRetro](https://github.com/catalystforyou/SimpRetro)) |
| `drugs_and_natural_products` | 21 | Drugs & natural products |

**Any dataset can be replaced** with your own targets by passing `--targets_file` (a `.csv` with a `SMILES` column, or a `.txt` with one SMILES per line):

```bash
python -m tree_search.run_mcts \
    --dataset_name custom \
    --targets_file /absolute/path/to/targets.csv \
    --model_name_list uspto_original_consol_Roh \
    --metric_name "sa_score&sc_score&ged"
```

### Metric-guided Retro* Search

The same chemistry metrics can also guide an A\*-style Retro\* search. This path uses the official **molstar** search core (`RSPlanner`) from [binghong-ml/retro_star](https://github.com/binghong-ml/retro_star), with the `template_relevance` (Roh) one-step adapter and local buyables wired to the same stack as the MCTS path. Here the chemistry score is fused into the one-step prior in log-space, `softmax(w·G + log p)` (`signed_rank_positive_only`), preserving the model policy `p`. This path reuses the MCTS metric, fusion, and schedule code.

#### One-time external setup

Retro\* depends on the external binghong clone **and** a small source patch (it wires the Roh one-step adapter, the metric-guidance arguments, and bounded-search depth into `RSPlanner`). The patch [`scripts/patches/binghong_retro_star.patch`](scripts/patches/binghong_retro_star.patch) is the reproducible record of the required changes. Run these once:

```bash
# 1. Clone the official Retro* (molstar) repo
git clone --depth 1 https://github.com/binghong-ml/retro_star.git external/binghong_retro_star

# 2. Download the official data assets (origin dict, one-step rollout model, value network) and unzip
wget -O retro_data.zip "https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=1"
unzip -o retro_data.zip -d external/binghong_retro_star/retro_star
rm retro_data.zip

# 3. Apply the metric-guidance patch
git -C external/binghong_retro_star apply scripts/patches/binghong_retro_star.patch

# 4. Install the two helper packages into the conda env
pip install -e external/binghong_retro_star/retro_star/packages/mlp_retrosyn
pip install -e external/binghong_retro_star/retro_star/packages/rdchiral
```

The run script puts the clone on `PYTHONPATH` automatically (via `BINGHONG_RETRO_STAR_ROOT`, default `external/binghong_retro_star`).

#### Run on USPTO-190

```bash
bash scripts/retro_star_uspto_190.sh smoke    # 2 targets, short budget (sanity check)
bash scripts/retro_star_uspto_190.sh all      # no-metric baseline + static weight + dynamic schedule
```

Or run a single configuration manually (omit `--metric_name` for the no-metric baseline):

```bash
python -m tree_search.run_retro_star \
    --dataset_name uspto_190 \
    --one_step_backend template_relevance \
    --retro_model_name uspto_original_consol_Roh \
    --metric_name "sa_score&sc_score&ged" \
    --metric_weight_start 0 \
    --metric_weight_end 1 \
    --metric_schedule_type exponential \
    --max_iterations 500 \
    --search_max_depth 10 \
    --expansion_topk 25 \
    --use_value_network 1 \
    --buyables_path tree_search/buyables.json.gz \
    --num_workers 10 \
    --pathway_output_folder outputs/uspto190_retro_star
```

Outputs follow the same `paths.json` / `stats.json` / `graph.json` layout under `--pathway_output_folder`.

### Optional: Integration with ASKCOS

The local modules can also be integrated into the full [ASKCOS framework](https://gitlab.com/mlpds_mit/askcosv2). To do so, follow the ASKCOS guide and point the code at ASKCOS HTTP APIs instead of the local ones:
- Replace `LocalExpandOneAPI` with `ExpandOneAPI` (pointing to the ASKCOS server)
- Replace `LocalPricerAPI` with `PricerAPI` (pointing to the ASKCOS server)

### Key Parameters for tree search

#### Dataset
- `--dataset_name`: `demo_targets`, `uspto_190`, `drug_hunter_172`, `drugs_and_natural_products`, or a custom name (with `--targets_file`)
- `--targets_file`: path to a custom targets file (`.csv` with a `SMILES` column, or `.txt` with one SMILES per line)
- `--model_name_list`: one-step model name(s), e.g. `uspto_original_consol_Roh`

#### Search (MCTS)
- `--max_iterations`: MCTS iterations per target (default: 100; 500 in the manuscript)
- `--max_depth`: maximum search tree depth (default: 10)
- `--num_workers`: number of parallel workers (default: 10 in the scripts)
- `--return_first`: stop a target once a route is found (1 = yes, 0 = no, default: 1)
- `--expansion_time`: maximum time per expansion in seconds (default: 12000)
- `--one_step_max_num_templates`: templates considered per expansion (default: 25)
- `--per_target_timeout`: wall-clock timeout per target in seconds (0 disables it)

#### Metric guidance
- `--metric_name`: any subset of `sa_score`, `sc_score`, `tanimoto`, `ged`, joined by `&`
- `--guidance_mode`: `model_prior_puct` (baseline) or `metric_hybrid_puct` (metric-guided)
- `--metric_aggregation_mode`: `rank_aggregation` or `signed_rank_positive_only`
- `--metric_weight_start` / `--metric_weight_end`: guidance-weight schedule bounds
- `--metric_schedule_type`: `constant`, `linear`, `exponential`, `logarithmic`, or `sigmoid`

#### Output
- `--pathway_output_folder`: output directory for results
- `--log_output`: log directory (default: `log_output`)

#### Retro*-specific (`run_retro_star`)
- `--one_step_backend`: `template_relevance` (Roh adapter, supports metric guidance) or `binghong_mlp`
- `--retro_model_name`: template-relevance model name (default: `uspto_original_consol_Roh`)
- `--search_max_depth`: bounded-search depth cap (e.g. `10`)
- `--use_value_network`: `1` to enable the Retro* value network, `0` for a zero value function
- `--expansion_topk`: one-step candidates per expansion
- `--buyables_path`: starting-molecules file (local `buyables.json.gz` or a Binghong CSV)
- `--gpu`: GPU index for one-step inference (`-1` for CPU)

## Acknowledgments

- ASKCOS framework for MCTS implementation: https://gitlab.com/mlpds_mit/askcosv2
- Retro\* (molstar) search core: https://github.com/binghong-ml/retro_star
- Data structures for synthesis tree and chemistry utilities: https://github.com/jihye-roh/higherlev_retro
- SCScore implementation: https://github.com/connorcoley/scscore
- RXNMapper for atom mapping: https://github.com/rxn4chemistry/rxnmapper
- Drug Hunter dataset (`drug_hunter_172`): https://github.com/catalystforyou/SimpRetro
