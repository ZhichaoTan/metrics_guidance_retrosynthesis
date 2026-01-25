# uspto_190
dataset_name="uspto_190"
model_name_list_list=("uspto_original_consol_Roh")
one_step_max_num_templates_list=(25)
ged_weight_start_end_list=("0 1")
return_first=1
ged_change_type_list=("logarithmic_change")
max_iteration_list=(500)
use_strategy_lib_flag_list=(0)
ged_output_folder="uspto_190_ged_folder_Oct_6"
pathway_folder="uspto_190_Oct_6"
deterministic_strategies_type="uspto_full"
metric_name_list=("SA_Score&SC_Score&tanimoto&ged")
ged_weight=999
max_depth_list=(10)
tanimoto_weight_in_metric=1
ged_change_type_in_metric="constant"
ged_weight_in_metric=0.02
use_external_temps=1
top_n_external_templates_list=(5 10 20 40)

for top_n_external_templates in "${top_n_external_templates_list[@]}"; do
        for use_strategy_lib_flag in "${use_strategy_lib_flag_list[@]}"; do
            for one_step_max_num_templates in "${one_step_max_num_templates_list[@]}"; do
                for max_iteration in "${max_iteration_list[@]}"; do
                    for model_name_list in "${model_name_list_list[@]}"; do
                        for ged_change_type in "${ged_change_type_list[@]}"; do
                            for ged_weight_start_end in "${ged_weight_start_end_list[@]}"; do
                                for metric_name in "${metric_name_list[@]}"; do
                                    for max_depth in "${max_depth_list[@]}"; do
                                        read -r ged_weight_start ged_weight_end <<< "$ged_weight_start_end"

                                        model_name_underscored=${model_name_list// /_}
                                        
                                        if [[ "$ged_change_type" == "constant" ]]; then
                                            ged_weight_start=$ged_weight
                                            ged_weight_end=$ged_weight
                                        fi

                                        echo "Using models: ${model_name_list}"
                                        echo "Max iteration: ${max_iteration}"
                                        echo "Change type: $ged_change_type, Start: $ged_weight_start, End: $ged_weight_end"
                                        echo "Return first: $return_first"
                                        echo "One-step max_num_templates: ${one_step_max_num_templates}"
                                        echo "Use strategy: ${use_strategy_lib_flag}"
                                        echo "Metric name: ${metric_name}"

                                        model_name_array=($model_name_list)
                                        python -m tree_search.ged_mcts_paral --dataset_name="$dataset_name" \
                                            --model_name_list "${model_name_array[@]}" \
                                            --ged \
                                            --ged_weight $ged_weight \
                                            --ged_weight_start $ged_weight_start \
                                            --ged_weight_end $ged_weight_end \
                                            --ged_change_type "$ged_change_type" \
                                            --pathway_output_folder "${pathway_folder}/pathway_folder_${model_name_underscored}_${max_iteration}_${top_n_external_templates}_${max_external_num}_${ged_weight}_metric_${metric_name}_return_first_${return_first}" \
                                            --num_worker 20 \
                                            --max_iterations "$max_iteration" \
                                            --return_first "$return_first" \
                                            --one_step_max_num_templates "$one_step_max_num_templates" \
                                            --strategy_max_num_templates 25 \
                                            --strategy_cut 1 \
                                            --use_strategy_lib "$use_strategy_lib_flag" \
                                            --track_ged_change 0 \
                                            --ged_output_folder "$ged_output_folder" \
                                            --deterministic_strategies_type "$deterministic_strategies_type" \
                                            --metric_name "$metric_name" \
                                            --max_depth "$max_depth" \
                                            --tanimoto_weight_in_metric "$tanimoto_weight_in_metric" \
                                            --ged_change_type_in_metric "$ged_change_type_in_metric" \
                                            --ged_weight_in_metric "$ged_weight_in_metric" \
                                            --use_external_temps $use_external_temps \
                                            --top_n_external_templates $top_n_external_templates
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    # done
done