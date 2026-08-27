#!/bin/bash
# Retro* on USPTO-190, with and without planner-native rank fusion (SA+SC+GED
# applied only when Retro* selects an open frontier node).
#
# Requires the external Retro* checkout (bash scripts/setup_binghong_retro_star.sh).
#
# Usage: bash scripts/retro_star_uspto_190.sh [baseline|guided|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

stage="${1:-all}"

MODEL_NAME="${MODEL_NAME:-uspto_original_consol_Roh}"
BUYABLES_PATH="${BUYABLES_PATH:-$PROJECT_ROOT/tree_search/buyables.json.gz}"
EXPANSION_TOPK="${EXPANSION_TOPK:-25}"
MAX_DEPTH="${MAX_DEPTH:-10}"
USE_VALUE_NETWORK="${USE_VALUE_NETWORK:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
NUM_WORKERS="${NUM_WORKERS:-10}"
GPU="${GPU:--1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uspto190_retro_star}"

BINGHONG_ROOT="${BINGHONG_RETRO_STAR_ROOT:-$PROJECT_ROOT/external/binghong_retro_star}"
export PYTHONPATH="$BINGHONG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BINGHONG_MLP_DEVICE="${BINGHONG_MLP_DEVICE:-$GPU}"

run_retro_star() {
    local guidance="$1"
    echo "== Retro* [guidance=$guidance] =="
    python -m tree_search.run_retro_star \
        --dataset_name uspto_190 \
        --pathway_output_folder "$OUTPUT_ROOT/$guidance" \
        --num_workers "$NUM_WORKERS" \
        --max_iterations "$MAX_ITERATIONS" \
        --expansion_topk "$EXPANSION_TOPK" \
        --buyables_path "$BUYABLES_PATH" \
        --one_step_backend template_relevance \
        --retro_model_name "$MODEL_NAME" \
        --use_value_network "$USE_VALUE_NETWORK" \
        --max_depth "$MAX_DEPTH" \
        --gpu "$GPU" \
        --guidance "$guidance"
}

case "$stage" in
    baseline) run_retro_star none ;;
    guided)   run_retro_star planner_native_rank_fusion ;;
    all)      run_retro_star none; run_retro_star planner_native_rank_fusion ;;
    *) echo "Usage: $0 [baseline|guided|all]"; exit 1 ;;
esac
