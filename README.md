# Metrics-Guided Retrosynthesis


### Software Dependencies

All dependencies are specified in `environment.yml`.

### ASKCOS Framework

## Installation Guide

### Step 1: Clone the Repository

```bash
cd metrics_guidance_retrosynthesis
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate metrics_guidance_retrosynthesis
```

### Step 3: Install ASKCOS Framework


## Demo Instructions

### Quick Start Demo

This demo uses a small subset (first 5 molecules) from the provided dataset to demonstrate the retrosynthesis search functionality.

### Step 1: Prepare Demo Dataset


### Step 2: Run Demo

```bash
python -m tree_search.ged_mcts_paral \
    --model_name_list uspto_original_consol_Roh \
    --ged_weight 0.5 \
    --max_iterations 100 \
    --num_workers 20 \
    --pathway_output_folder demo_output
```

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
