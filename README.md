# Guiding Computer-aided Multistep Retrosynthesis with Complexity and Distance Metrics

This repository contains the implementation for integrating complexity and distance metrics to effectively guide multistep retrosynthesis using Monte Carlo Tree Search (MCTS).

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Code Structure](#code-structure)
- [Guiding Metric Calculation](#guiding-metric-calculation)
- [Metric-guided Retrosynthetic Search](#metric-guided-retrosynthetic-search)
- [Optional: Integration with ASKCOS](#optional-integration-with-askcos)
- [Usage Instructions](#usage-instructions)
- [Key Parameters](#key-parameters)
- [Demo Instructions](#demo-instructions)
- [Reproduction Instructions](#reproduction-instructions)
- [Troubleshooting](#troubleshooting)

## Overview

This implementation consists of two core modules:

1. **Guiding metric calculation**: Automates the calculation of graph edit distance (GED) between intermediates and the target molecule, following the manual approach described in the [Science paper](https://www.science.org/doi/10.1126/science.ade8459). GED calculation supports both reaction-level ([`GED_scorer.py`](metrics/GED_scorer.py)) and pathway-level ([`syn_tree_ged.py`](metrics/syn_tree_ged.py)) measurements. This implementation also supports other complexity metrics (SAscore and SCScore) and distance metrics (Tanimoto distance).

2. **Metric-guided retrosynthetic search**: Integrates metric calculations into the Monte Carlo Tree Search (MCTS) framework. The one-step model is consistent with [higherlev_retro](https://github.com/jihye-roh/higherlev_retro). **This implementation supports local deployment** - all models and APIs run directly on your machine without requiring Docker or external servers. We adapt corresponding modules from [ASKCOS](https://gitlab.com/mlpds_mit/askcosv2) for local deployment, but these modules can also be integrated into ASKCOS.

## System Requirements

The codebase has been developed and tested primarily on Linux (Ubuntu 18.04/20.04), but should also work on macOS and Windows with compatible environments.

**Recommended system requirements:**
- **Operating System:** Linux (Ubuntu 18.04/20.04), macOS, or Windows 10/11
- **RAM:** 32 GB or higher (16 GB minimum for small-scale testing)
- **CPU:** Multi-core processor (at least 10 cores recommended for parallel multistep retrosynthesis)
- **Optional:** NVIDIA GPU (with CUDA for accelerating one-step model inference)
- **Python:** 3.8-3.10 (tested with 3.10)

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

Download the `uspto_original_consol.mar` file using the data download link provided in the [higherlev_retro repository](https://github.com/jihye-roh/higherlev_retro). To extract the contents of the `.mar` file (which is a zip archive), use the following command in your terminal:

```bash
unzip uspto_original_consol.mar -d tree_search/uspto_original_consol_Roh
```

This will unpack the model files into the `tree_search/uspto_original_consol_Roh/` directory.

After extraction, the directory should contain:
- `model_latest.pt` - The trained model weights
- `templates.jsonl` - Template library
- `utils.py` - Model utilities
- `models.py` - Model architecture definitions
- `misc.py` - Miscellaneous utilities

## Code Structure

```
metrics_guidance_retrosynthesis/
├── tree_search/                    # Main retrosynthesis search module
│   ├── ged_mcts_paral.py          # Parallel MCTS execution script
│   ├── mcts/                       # MCTS implementation
│   │   ├── mcts_controller.py     # Base MCTS controller
│   │   ├── mcts_ged_controller.py # Metrics-guided MCTS controller
│   │   ├── options.py             # Configuration options
│   │   ├── ged_weight_change.py  # GED weight scheduling functions
│   │   └── modules/               # Local API modules
│   │       ├── local_expand_one_api.py      # Local one-step expansion API
│   │       ├── local_pricer_api.py          # Local buyables pricer API
│   │       ├── local_retro_controller.py    # Local retrosynthesis controller
│   │       ├── local_templ_rel_predictor.py # Template relevance predictor
│   │       └── expand_one_controller.py    # One-step expansion controller
│   ├── buyables.json.gz           # Buyable molecules database from ASKCOS
│   └── uspto_original_consol_Roh/ # Retrosynthesis model directory
├── metrics/                        # Metric calculation modules
│   ├── GED_scorer.py              # Reaction-level GED calculation
│   ├── syn_tree_ged.py            # Pathway-level GED calculation
│   ├── sascore.py                 # SA_Score calculation
│   └── scscore.py                 # SC_Score calculation
├── data/                          # Dataset files
│   ├── demo_targets.csv           # Demo dataset
│   └── uspto_190_targets.csv      # Full USPTO-190 dataset
├── scripts/                       # Utility scripts
│   └── uspto_190.sh              # Batch processing script
└── utils/                         # Utility functions
    └── chem_utils.py             # Chemistry utilities
```

## Guiding Metric Calculation

This module provides automated calculation of various metrics to guide retrosynthetic search, including complexity metrics (SA_Score, SC_Score) and distance metrics (Tanimoto distance, Graph Edit Distance).

### Existing Guiding Metrics

**SA_Score** and **Tanimoto distance** are directly implemented using RDKit. **SC_Score** is calculated using the [reference repository](https://github.com/connorcoley/scscore).

### Graph Edit Distance (GED)

We implement GED calculation at two levels: reaction-level and pathway-level. GED measures the number of bond and/or chirality edits required to transform an intermediate into the target molecule.

#### Reaction-level GED

The reaction-level GED quantifies the structural change occurring in a single reaction step. It is implemented in [`GED_scorer.py`](metrics/GED_scorer.py). 

**Important**: Atoms between reactants and products must be mapped before input. We recommend using [RXNMapper](https://github.com/rxn4chemistry/rxnmapper) for automatic atom mapping.

**Example usage:**

```python
from metrics.GED_scorer import GED_scorer

# Example reaction SMILES with atom mapping
reaction_smiles = "[C:11](=[O:12])([CH2:13][CH2:14][CH:19]=[O:20])[OH:18].[CH3:1][C@H:2]([CH2:3][OH:4])[CH2:16][CH:6]=[O:5].[NH2:10][CH3:15]>>[CH3:1][C@H:2]([CH2:3][OH:4])[C@H:16]([CH:6]=[O:5])[C@H:15]([NH2:10])[CH2:14][CH2:13][C:11](=[O:12])[OH:18]"

# Calculate GED for a single reaction
scorer = GED_scorer(reaction_smiles)
score = scorer.calculate_score()
print("GED score for this reaction:", score)
```

#### Pathway-level GED

The pathway-level GED evaluates the cumulative structural changes across an entire synthetic pathway. It is implemented in [`syn_tree_ged.py`](metrics/syn_tree_ged.py), which extends the synthesis tree data structure with GED calculation capabilities.

**Features:**
- Automatic atom mapping for unmapped reactions using RXNMapper
- Tracking of GED changes along the synthesis pathway
- Visualization of metric trajectories

**Example usage:**

```python
from metrics.syn_tree_ged import SynTree_ged

# Example retrosynthetic route
# If reactions are not atom-mapped, they will be automatically mapped using RXNMapper
example_route = [
    "C.C=CCC=O.O>>[H]C(=O)C[C@H](C)CO",
    "COC1=CC=C(C=C1)N.O=CCCC(=O)O.[H]C(=O)C[C@H](C)CO>>[H]C(=O)[C@]([H])([C@H](C)CO)[C@@H](N)CCC(=O)O",
    "C=CC.[H]C(=O)[C@]([H])([C@H](C)CO)[C@@H](N)CCC(=O)O>>[H][C@]([C@H](C)CO)([C@H](O)CC=C)[C@@]([H])(N)CCC(=O)O"
]

# Create synthesis tree with GED tracking
syn_tree = SynTree_ged(route=example_route, track_atom_init=True, add_ged_init=True)

# Visualize metric changes along the pathway
syn_tree.visualize_pathway_metric_changes(output_number=False)
```

## Metric-guided Retrosynthetic Search

This module integrates metric calculations into the Monte Carlo Tree Search (MCTS) framework to guide the retrosynthetic exploration process. The implementation uses local deployment, meaning all components run directly on your machine without requiring Docker or external servers.

## Optional: Integration with ASKCOS

While this implementation provides local deployment for standalone use, the modules can also be integrated into the full [ASKCOS framework](https://gitlab.com/mlpds_mit/askcosv2) for more comprehensive retrosynthesis capabilities.

### Integration Steps

To integrate with ASKCOS:

1. **Deploy ASKCOS**: Follow the [ASKCOS guide](https://gitlab.com/mlpds_mit/askcosv2)

2. **Replace local APIs**: Modify the code to use ASKCOS HTTP APIs instead of local APIs:
   - Replace `LocalExpandOneAPI` with `ExpandOneAPI` (pointing to ASKCOS server)
   - Replace `LocalPricerAPI` with `PricerAPI` (pointing to ASKCOS server)

## Usage Instructions

### Basic Usage

Run retrosynthesis search on a dataset:

```bash
python -m tree_search.ged_mcts_paral     --dataset_name demo_targets     --model_name_list uspto_original_consol_Roh     --metric_name ged     --ged_weight 0.5     --max_iterations 100     --num_workers 5     --pathway_output_folder demo_output
```

### Running from Script Directory

The script automatically detects the project root, but for best results, run from the project root directory:

```bash
cd /path/to/metrics_guidance_retrosynthesis
python -m tree_search.ged_mcts_paral [arguments...]
```

## Key Parameters

### Dataset Parameters
- `--dataset_name`: Name identifier for your dataset
  - `demo_targets`: Small demo dataset (5 molecules)
  - `uspto_190`: Full USPTO-190 dataset (190 molecules)
- `--model_name_list`: List of model names (e.g., `uspto_original_consol_Roh`)

### MCTS Search Parameters
- `--max_iterations`: Maximum MCTS iterations per target (default: 100)
- `--max_depth`: Maximum search tree depth (default: 10)
- `--num_workers`: Number of parallel workers (recommended: number of CPU cores, default: 20)
- `--return_first`: Whether to stop search once a pathway is found (1 = yes, 0 = no, default: 1)
- `--expansion_time`: Maximum time per expansion in seconds (default: 12000)

### Metric Parameters
- `--metric_name`: Metric to use for guidance
  - Single metrics: `ged`, `tanimoto`, `SA_Score`, `SC_Score`
  - Hybrid metrics: `ged&tanimoto`, `SA_Score&SC_Score&tanimoto&ged` (use `&` to combine)
- `--ged_weight`: Weight for the metric in UCB scoring (0.0 to 1.0, default: 0)
- `--ged_weight_start`: Starting weight for dynamic weight adjustment (default: 0)
- `--ged_weight_end`: Ending weight for dynamic weight adjustment (default: 1)
- `--ged_change_type`: Type of metric weight change function:
  - `constant`: Constant weight throughout search
  - `linear`: Linear interpolation from start to end weight
  - `logarithmic_change`: Logarithmic change over iterations
  - `exponential_change`: Exponential change over iterations
  - `hybrid_exponential_change`: Hybrid exponential change

### Hybrid Metric Parameters
- `--ged_weight_in_metric`: Weight for GED in hybrid metrics (default: 0)
- `--tanimoto_weight_in_metric`: Weight for Tanimoto in hybrid metrics (default: 0)
- `--ged_change_type_in_metric`: Change type for GED weight within hybrid metric (default: `constant`)

### One-Step Model Parameters
- `--one_step_max_num_templates`: Maximum templates to consider per expansion (default: 25)

### Output Parameters
- `--pathway_output_folder`: Output directory for results (default: `pathway_output_folder`)
- `--log_output`: Log folder (default: `log_output`)

## Demo Instructions

### Quick Start Demo

This demo uses a small subset (first 5 molecules) from the provided dataset to demonstrate the retrosynthesis search functionality.

### Step 1: Prepare Demo Dataset

The demo dataset is already included in `data/demo_targets.csv`.

### Step 2: Run Demo

```bash
python -m tree_search.ged_mcts_paral     --dataset_name demo_targets     --model_name_list uspto_original_consol_Roh     --metric_name ged     --ged_weight 0.5     --max_iterations 100     --num_workers 5     --pathway_output_folder demo_output
```

### Step 3: Check Results

Results will be saved in `demo_output/` with one subdirectory per molecule containing:
- `paths.json`: Found retrosynthetic pathways
- `stats.json`: Search statistics
- `graph.json`: Search tree graph

Logs will be saved in `log_output/` with detailed information about the search process.

## Reproduction Instructions

To reproduce the quantitative results from the manuscript:

### Step 1: Use the Provided Dataset

The full dataset `data/uspto_190_targets.csv` contains 190 target molecules used in the study.

### Step 2: Run with Manuscript Parameters

Use the provided script `scripts/uspto_190.sh` which contains the exact parameters used in the manuscript:

```bash
bash scripts/uspto_190.sh
```

Or run manually with the following parameters:

```bash
python -m tree_search.ged_mcts_paral     --dataset_name uspto_190     --model_name_list uspto_original_consol_Roh     --metric_name "SA_Score&SC_Score&tanimoto&ged"     --ged_weight 999     --ged_weight_start 0     --ged_weight_end 1     --ged_change_type logarithmic_change     --ged_weight_in_metric 0.02     --tanimoto_weight_in_metric 1     --max_iterations 500     --max_depth 10     --one_step_max_num_templates 25     --num_workers 20     --pathway_output_folder uspto_190_results
```

## Acknowledgments

- ASKCOS framework for MCTS implementation: https://gitlab.com/mlpds_mit/askcosv2
- Data structures for synthesis tree and more: https://github.com/jihye-roh/higherlev_retro
- SCScore implementation: https://github.com/connorcoley/scscore
- RXNMapper for atom mapping: https://github.com/rxn4chemistry/rxnmapper