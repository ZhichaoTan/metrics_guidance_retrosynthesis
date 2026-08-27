#!/bin/bash
# Small single-process demo: metric-guided MCTS (core setting) on the 5 bundled demo targets,
# one worker, 100 iterations. A quick end-to-end sanity check.
#
# Usage: bash scripts/demo_single_process.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

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
    --pathway_output_folder demo_output
