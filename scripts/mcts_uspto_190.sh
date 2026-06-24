#!/bin/bash
# Metric-guided MCTS on USPTO-190: metric-hybrid PUCT with the sa_score&sc_score&ged
# chemistry guidance aggregated by signed-rank and ramped on a logarithmic schedule.
# Override any setting via environment variables, e.g. NUM_WORKERS=16 bash scripts/mcts_uspto_190.sh
#
# Usage: bash scripts/mcts_uspto_190.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

DATASET_NAME="${DATASET_NAME:-uspto_190}"
MODEL_NAME="${MODEL_NAME:-uspto_original_consol_Roh}"
METRIC_NAME="${METRIC_NAME:-sa_score&sc_score&ged}"
GUIDANCE_MODE="${GUIDANCE_MODE:-metric_hybrid_puct}"
AGGREGATION_MODE="${AGGREGATION_MODE:-signed_rank_positive_only}"
SCHEDULE="${SCHEDULE:-logarithmic}"
METRIC_WEIGHT_START="${METRIC_WEIGHT_START:-0}"
METRIC_WEIGHT_END="${METRIC_WEIGHT_END:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
MAX_DEPTH="${MAX_DEPTH:-10}"
NUM_WORKERS="${NUM_WORKERS:-10}"
ONE_STEP_MAX_NUM_TEMPLATES="${ONE_STEP_MAX_NUM_TEMPLATES:-25}"
PER_TARGET_TIMEOUT="${PER_TARGET_TIMEOUT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uspto190_core_mcts}"

python -m tree_search.run_mcts \
    --dataset_name "$DATASET_NAME" \
    --model_name_list "$MODEL_NAME" \
    --metric_name "$METRIC_NAME" \
    --guidance_mode "$GUIDANCE_MODE" \
    --metric_aggregation_mode "$AGGREGATION_MODE" \
    --metric_weight_start "$METRIC_WEIGHT_START" \
    --metric_weight_end "$METRIC_WEIGHT_END" \
    --metric_schedule_type "$SCHEDULE" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_depth "$MAX_DEPTH" \
    --one_step_max_num_templates "$ONE_STEP_MAX_NUM_TEMPLATES" \
    --num_workers "$NUM_WORKERS" \
    --per_target_timeout "$PER_TARGET_TIMEOUT" \
    --pathway_output_folder "$OUTPUT_ROOT"
