#!/bin/bash
# Metric-guided Retro* on USPTO-190: inject the signed-rank chemistry score
# (sa_score&sc_score&ged) into the binghong RSPlanner via the template_relevance (Roh) one-step
# adapter. The same entry point runs the no-metric baseline. Requires the external binghong assets
# (see the "Metric-guided Retro* Search" setup steps in README.md, or set BINGHONG_RETRO_STAR_ROOT).
#
# Usage: bash scripts/retro_star_uspto_190.sh [smoke|baseline|static|dynamic|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

stage="${1:-all}"

DATASET_NAME="uspto_190"
MODEL_NAME="${MODEL_NAME:-uspto_original_consol_Roh}"
BUYABLES_PATH="${BUYABLES_PATH:-$PROJECT_ROOT/tree_search/buyables.json.gz}"
METRIC_NAME="${METRIC_NAME:-sa_score&sc_score&ged}"
SCHEDULE="${SCHEDULE:-exponential}"
STATIC_BETA="${STATIC_BETA:-0.7}"
BETA_START="${BETA_START:-0}"
BETA_END="${BETA_END:-1}"
EXPANSION_TOPK="${EXPANSION_TOPK:-25}"
SEARCH_MAX_DEPTH="${SEARCH_MAX_DEPTH:-10}"
USE_VALUE_NETWORK="${USE_VALUE_NETWORK:-1}"
GPU="${GPU:--1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
SMOKE_ITERATIONS="${SMOKE_ITERATIONS:-5}"
NUM_WORKERS="${NUM_WORKERS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uspto190_retro_star}"
LOG_OUTPUT="${LOG_OUTPUT:-log_output}"

BINGHONG_ROOT="${BINGHONG_RETRO_STAR_ROOT:-$PROJECT_ROOT/external/binghong_retro_star}"
export PYTHONPATH="$BINGHONG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BINGHONG_MLP_DEVICE="${BINGHONG_MLP_DEVICE:-$GPU}"

run_retro_star() {
    local folder="$1" m_name="$2" beta_start="$3" beta_end="$4" schedule="$5"
    local iterations="${6:-$MAX_ITERATIONS}" targets_file="${7:-}"

    local output_folder="$OUTPUT_ROOT/$folder"
    local log_folder="$LOG_OUTPUT/uspto_190/retro_star/$folder"
    mkdir -p "$output_folder" "$log_folder"

    local cmd=(
        python -m tree_search.run_retro_star
        --dataset_name "$DATASET_NAME"
        --log_output "$log_folder"
        --pathway_output_folder "$output_folder"
        --num_workers "$NUM_WORKERS"
        --max_iterations "$iterations"
        --expansion_topk "$EXPANSION_TOPK"
        --buyables_path "$BUYABLES_PATH"
        --one_step_backend template_relevance
        --retro_model_name "$MODEL_NAME"
        --use_value_network "$USE_VALUE_NETWORK"
        --gpu "$GPU"
        --search_max_depth "$SEARCH_MAX_DEPTH"
    )
    if [[ -n "$m_name" ]]; then
        cmd+=(--metric_name "$m_name" --metric_weight_start "$beta_start"
              --metric_weight_end "$beta_end" --metric_schedule_type "$schedule")
    fi
    if [[ -n "$targets_file" ]]; then
        cmd+=(--targets_file "$targets_file")
    fi

    echo "== Retro* [$folder] metric=${m_name:-<none>} beta=$beta_start->$beta_end ($schedule) iter=$iterations =="
    "${cmd[@]}"
}

run_baseline() {
    run_retro_star "baseline_vn${USE_VALUE_NETWORK}_d${SEARCH_MAX_DEPTH}_topk${EXPANSION_TOPK}" \
        "" 0 0 constant
}
run_static() {
    run_retro_star "metric_static_b${STATIC_BETA}_vn${USE_VALUE_NETWORK}_d${SEARCH_MAX_DEPTH}_topk${EXPANSION_TOPK}" \
        "$METRIC_NAME" "$STATIC_BETA" "$STATIC_BETA" constant
}
run_dynamic() {
    run_retro_star "metric_dynamic_${SCHEDULE}_vn${USE_VALUE_NETWORK}_d${SEARCH_MAX_DEPTH}_topk${EXPANSION_TOPK}" \
        "$METRIC_NAME" "$BETA_START" "$BETA_END" "$SCHEDULE"
}
run_smoke() {
    local smoke_dir="${TMPDIR:-/tmp}/retro_star_smoke_$$"
    mkdir -p "$smoke_dir"
    python - "$smoke_dir" "$DATASET_NAME" <<'PY'
import sys
from pathlib import Path
from tree_search.experiment_utils import load_targets
smoke_dir = Path(sys.argv[1]); smoke_dir.mkdir(parents=True, exist_ok=True)
targets = load_targets(sys.argv[2])
if not targets:
    raise SystemExit("No targets loaded")
(smoke_dir / "two.txt").write_text("\n".join(targets[:2]) + "\n", encoding="utf-8")
PY
    run_retro_star "smoke_metric_static_iter${SMOKE_ITERATIONS}" \
        "$METRIC_NAME" "$STATIC_BETA" "$STATIC_BETA" constant "$SMOKE_ITERATIONS" "$smoke_dir/two.txt"
}

case "$stage" in
    smoke)    run_smoke ;;
    baseline) run_baseline ;;
    static)   run_static ;;
    dynamic)  run_dynamic ;;
    all)      run_baseline; run_static; run_dynamic ;;
    *) echo "Usage: $0 [smoke|baseline|static|dynamic|all]"; exit 1 ;;
esac
