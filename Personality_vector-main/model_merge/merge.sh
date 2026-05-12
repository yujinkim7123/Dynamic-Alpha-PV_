#!/bin/bash

BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

FINETUNED_MODEL_PATHS=(
  ""
)

OUTPUT_MODEL_PATH="./merged_HIGH_AGR_LOW_AGR"

# (D) choices: [average_merging, task_arithmetic, mask_merging, fisher_merging,
#               regmean_merging, ties_merging]
MERGING_METHOD_NAME="task_arithmetic"

# (E) choices: [average_merging, task_arithmetic, fisher_merging,
#               regmean_merging, ties_merging]
MASK_APPLY_METHOD="task_arithmetic"

# (F) DaRE parameter
WEIGHT_FORMAT="delta_weight"    # or "finetuned_weight"
WEIGHT_MASK_RATE=0.5          
USE_WEIGHT_RESCALE="--use_weight_rescale"           
MASK_STRATEGY="random"          # or "magnitude"

# (G) Task Arithmetic scale
SCALING_COEFFICIENT=0.7

# (H) EXCLUDE_PARAMAMETER
EXCLUDE_PARAM_REGEX=""

# (I) Seed
SEED=42

# (J) indicidual scaling for multiple task vector
INDIVIDUAL_SCALING="true"   # 또는 "false"

# (K) individual scaling coefficients. Only with INDIVIDUAL_SCALING="true"
SCALING_COEFFICIENTS=(0.4 1.6)

############################
# (2) Run
############################

FINETUNED_MODEL_PATHS_STR="${FINETUNED_MODEL_PATHS[@]}"

if [ -n "$USE_WEIGHT_RESCALE" ]; then
  USE_WEIGHT_RESCALE="--use_weight_rescale"
fi

EXCLUDE_PARAM_ARG=""
if [ -n "$EXCLUDE_PARAM_REGEX" ]; then
  EXCLUDE_PARAM_ARG="--exclude_param_names_regex $EXCLUDE_PARAM_REGEX"
fi

echo "======================================"
echo "[merge.sh] Base Model: $BASE_MODEL_PATH"
echo "[merge.sh] Finetuned Models: ${FINETUNED_MODEL_PATHS_STR}"
echo "[merge.sh] Output Path: $OUTPUT_MODEL_PATH"
echo "[merge.sh] Merging Method: $MERGING_METHOD_NAME"
echo "[merge.sh] Mask Apply Method: $MASK_APPLY_METHOD"
echo "[merge.sh] Weight Format: $WEIGHT_FORMAT"
echo "[merge.sh] Weight Mask Rate: $WEIGHT_MASK_RATE"
echo "[merge.sh] Use Weight Rescale? $USE_WEIGHT_RESCALE"
echo "[merge.sh] Mask Strategy: $MASK_STRATEGY"
echo "[merge.sh] Scaling Coefficient: $SCALING_COEFFICIENT"
echo "[merge.sh] Exclude Param Regex: $EXCLUDE_PARAM_REGEX"
echo "[merge.sh] Seed: $SEED"
echo "======================================"

python3 merge.py \
  --base_model_path "$BASE_MODEL_PATH" \
  --finetuned_model_paths ${FINETUNED_MODEL_PATHS[@]} \
  --output_model_path "$OUTPUT_MODEL_PATH" \
  --merging_method_name "$MERGING_METHOD_NAME" \
  --mask_apply_method "$MASK_APPLY_METHOD" \
  --weight_format "$WEIGHT_FORMAT" \
  --weight_mask_rate "$WEIGHT_MASK_RATE" \
  $USE_WEIGHT_RESCALE \
  --mask_strategy "$MASK_STRATEGY" \
  --scaling_coefficient "$SCALING_COEFFICIENT" \
  --individual_scaling "$INDIVIDUAL_SCALING" \
  --scaling_coefficients ${SCALING_COEFFICIENTS[@]} \
  $EXCLUDE_PARAM_ARG \
  --seed $SEED