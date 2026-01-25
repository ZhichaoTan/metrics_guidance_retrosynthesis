# Metrics-Guided Retrosynthesis

This repo contains the code for Integrating Complexity and Distance Metrics to Effectively Guide Multistep Retrosynthesis.

## System Requirements

### Hardware Requirements
- **CPU**: 20 cores (multi-core processing recommended to parallel multistep searches)
- **GPU**: Optional (not required for basic functionality but usable in one-step model inference in ASKCOS)

### Software Dependencies

All dependencies are specified in `environment.yml`.

### ASKCOS Framework
For full ASKCOS deployment and documentation, please refer to the [ASKCOSv2 GitLab repository](https://gitlab.com/mlpds_mit/askcosv2). The one-step model and tree search modules are modified for multistep retrosynthesis. 

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

### Step 3: Install ASKCOS Framework

Please refer to the [ASKCOSv2 GitLab repository](https://gitlab.com/mlpds_mit/askcosv2) for complete ASKCOS installation and deployment instructions.

## Demo Instructions

### Quick Start Demo

This demo uses a small subset (first 5 molecules) from the provided dataset to demonstrate the retrosynthesis search functionality.

### Step 1: Prepare Demo Dataset

The demo dataset is already included in `data/demo_targets.csv`.

### Step 2: Run Demo

```bash
python -m tree_search.ged_mcts_paral \
    --dataset_name demo_targets \
    --model_name_list uspto_original_consol_Roh \
    --ged_weight 0.5 \
    --max_iterations 100 \
    --num_workers 20 \
    --pathway_output_folder demo_output
```

### Step 3: Check Output

The demo will generate output files in the `demo_output/` directory:
- `paths.json`: Found retrosynthetic pathways
- `stats.json`: Search statistics
- `graph.json`: Search tree graph structure

### Expected Output

The demo should complete successfully and produce:
- **Output location**: `demo_output/demo_targets/` directory
- **Output files**: 
  - `paths.json`: List of retrosynthetic pathways found
  - `stats.json`: Search statistics including success rate, iterations, etc.
  - `graph.json`: Complete search tree structure
- **Expected runtime**: 5-15 minutes on a system with 20 CPU cores

## Instructions for Use

### Key Parameters

- `--dataset_name`: Name identifier for your dataset (use `demo_targets` for demo dataset, `uspto_190` for full dataset)
- `--model_name_list`: List of ASKCOS model names (e.g., `uspto_original_consol_Roh`)
- `--ged_weight`: Weight for GED metric in UCB scoring (0.0 to 1.0)
- `--ged_weight_start`: Starting weight for dynamic GED weight adjustment (default: 0)
- `--ged_weight_end`: Ending weight for dynamic GED weight adjustment (default: 1)
- `--ged_change_type`: Type of GED weight change function:
  - `constant`: Constant weight throughout search
  - `linear`: Linear interpolation from start to end weight
  - `logarithmic_change`: Logarithmic change over iterations
  - `exponential_change`: Exponential change over iterations
  - `hybrid_exponential_change`: Hybrid exponential change
- `--max_iterations`: Maximum MCTS iterations per target
- `--num_workers`: Number of parallel workers (recommended: number of CPU cores, default: 20)
- `--pathway_output_folder`: Output directory for results
- `--max_depth`: Maximum search tree depth
- `--one_step_max_num_templates`: Maximum templates to consider per expansion
- `--metric_name`: Metric to use (`ged`, `tanimoto`, `SA_Score`, `SC_Score`, or hybrid like `ged&tanimoto`)
- `--ged_weight_in_metric`: Weight for GED in hybrid metrics
- `--tanimoto_weight_in_metric`: Weight for Tanimoto in hybrid metrics

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
python -m tree_search.ged_mcts_paral \
    --dataset_name uspto_190 \
    --model_name_list uspto_original_consol_Roh \
    --metric_name "SA_Score&SC_Score&tanimoto&ged" \
    --ged_weight 999 \
    --ged_weight_start 0 \
    --ged_weight_end 1 \
    --ged_change_type logarithmic_change \
    --ged_weight_in_metric 0.02 \
    --tanimoto_weight_in_metric 1 \
    --max_iterations 500 \
    --max_depth 10 \
    --one_step_max_num_templates 25 \
    --num_workers 20 \
    --pathway_output_folder uspto_190_results
```
## Acknowledgments
- ASKCOS framework for MCTS implementation https://gitlab.com/mlpds_mit/askcosv2
- Data structures for synthesis tree and more https://github.com/jihye-roh/higherlev_retro