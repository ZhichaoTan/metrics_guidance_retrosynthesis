#!/bin/bash
# Run from metrics_guidance_retrosynthesis directory
# Get script directory and change to project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Configuration
dataset_name="uspto_190"
model_name="uspto_original_consol_Roh"
ged_weight_start=0
ged_weight_end=1
ged_change_type="logarithmic_change"
max_iteration=100
deterministic_strategies_type="uspto_full"
metric_name="SA_Score&SC_Score&tanimoto&ged"
tanimoto_weight_in_metric=1
ged_weight_in_metric=0.02
model_name_underscored=${model_name// /_}
num_workers=10
pathway_folder="uspto_190_pathway_folder"

if [[ "$ged_change_type" == "constant" ]]; then
    ged_weight_start=$ged_weight
    ged_weight_end=$ged_weight
fi

echo "=========================================="
echo "Using model: ${model_name}"
echo "Max iteration: ${max_iteration}"
echo "Change type: $ged_change_type, Start: $ged_weight_start, End: $ged_weight_end"
echo "Use strategy: ${use_strategy_lib_flag}"
echo "Metric name: ${metric_name}"
echo "Top N external templates: ${top_n_external_templates}"
echo "=========================================="

python -m tree_search.ged_mcts_paral \
    --dataset_name="$dataset_name" \
    --model_name_list "$model_name" \
    --ged_weight_start $ged_weight_start \
    --ged_weight_end $ged_weight_end \
    --ged_change_type "$ged_change_type" \
    --pathway_output_folder "${pathway_folder}/${model_name_underscored}_${max_iteration}_${ged_weight}_metric_${metric_name}_return_first_1" \
    --num_workers $num_workers \
    --max_iterations "$max_iteration" \
    --metric_name "$metric_name" \
    --tanimoto_weight_in_metric "$tanimoto_weight_in_metric" \
    --ged_weight_in_metric "$ged_weight_in_metric"
