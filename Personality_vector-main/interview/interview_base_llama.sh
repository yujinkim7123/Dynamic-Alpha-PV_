#!/bin/bash
AGENT_TYPE="mycustom"
QUESTIONNAIRE_NAME="BFI"       # Big Five
CHARACTER="myagent"            
AGENT_LLM="gpt-3.5-turbo"
EVALUATOR_LLM="gpt-4o"
EVAL_METHOD="interview_batch"
LLAMA_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"  
REPEAT_TIMES=1

RESULT_DIR="../results/final/${QUESTIONNAIRE_NAME}_agent-type=${AGENT_TYPE}_agent-llm=${AGENT_LLM}_eval-method=${EVAL_METHOD}-${EVALUATOR_LLM}_repeat-times=${REPEAT_TIMES}"

mkdir -p "$RESULT_DIR"


# 1) Extraversion (EXT)
EXT_LOW=("unfriendly" "silent" "inactive" "timid" "unassertive" "unenergetic" "unadventurous" "gloomy")
EXT_HIGH=("friendly"  "talkative" "energetic" "adventurous and daring" "cheerful" "bold" "assertive" "active")

# 2) Agreeableness (AGR)
AGR_LOW=("distrustful" "immoral" "dishonest" "unkind" "stingy" "unaltruistic" "uncooperative" "self-important" "unsympathetic" "selfish")
AGR_HIGH=("trustful" "moral" "honest" "kind" "generous" "altruistic" "cooperative" "humble" "sympathetic" "unselfish")

# 3) Conscientiousness (CON)
CON_LOW=("unsure" "messy" "irresponsible" "lazy" "undisciplined" "impractical" "extravagant" "disorganized" "negligent" "careless")
CON_HIGH=("self-efficacious" "orderly" "responsible" "hardworking" "self-disciplined" "practical" "thrifty" "organized" "thorough")

# 4) Neuroticism (NEU)
NEU_LOW=("relaxed" "at ease" "easygoing" "calm" "patient" "happy" "unselfconscious" "level-headed" "contented" "emotionally stable")
NEU_HIGH=("tense" "nervous" "anxious" "angry" "irritable" "depressed" "self-conscious" "impulsive" "discontented" "emotionally unstable")

# 5) Openness (OPN)
OPN_LOW=("unimaginative" "uncreative" "artistically unappreciative" "unaesthetic" "unreflective" "emotionally closed" "uninquisitive" "predictable" "unintelligent" "unanalytical" "unsophisticated" "socially conservative")
OPN_HIGH=("imaginative" "creative" "artistically appreciative" "aesthetic" "reflective" "emotionally aware" "curious" "spontaneous" "intelligent" "analytical" "sophisticated" "socially progressive")


apply_scale_prefixes() {
  local scale="$1"
  shift
  local facets=("$@")  # 

  local prefix=""
  case "$scale" in
    1) prefix="a bit" ;;
    2) prefix="" ;;
    3) prefix="very" ;;
    *) prefix="" ;;
  esac

  local result=""
  for facet in "${facets[@]}"; do
    if [ -n "$prefix" ]; then
      result="$result, $prefix $facet"
    else
      result="$result, $facet"
    fi
  done

  result="${result#, }"
  echo "$result"
}


POLARITY_LIST=("HIGH" "LOW")
TRAIT_LIST=("AGR" "CON" "EXT" "NEU" "OPN")
SCALE_LIST=(1 2 3)

for polarity in "${POLARITY_LIST[@]}"; do
  for trait in "${TRAIT_LIST[@]}"; do
    for scale in "${SCALE_LIST[@]}"; do

      case "$trait" in
        "EXT")
          if [ "$polarity" = "HIGH" ]; then
            trait_array=("${EXT_HIGH[@]}")
          else
            trait_array=("${EXT_LOW[@]}")
          fi
          ;;
        "AGR")
          if [ "$polarity" = "HIGH" ]; then
            trait_array=("${AGR_HIGH[@]}")
          else
            trait_array=("${AGR_LOW[@]}")
          fi
          ;;
        "CON")
          if [ "$polarity" = "HIGH" ]; then
            trait_array=("${CON_HIGH[@]}")
          else
            trait_array=("${CON_LOW[@]}")
          fi
          ;;
        "NEU")
          if [ "$polarity" = "HIGH" ]; then
            trait_array=("${NEU_HIGH[@]}")
          else
            trait_array=("${NEU_LOW[@]}")
          fi
          ;;
        "OPN")
          if [ "$polarity" = "HIGH" ]; then
            trait_array=("${OPN_HIGH[@]}")
          else
            trait_array=("${OPN_LOW[@]}")
          fi
          ;;
      esac

      random_five=($(printf "%s\n" "${trait_array[@]}" | shuf -n 5))

      final_adverbs="$(apply_scale_prefixes "$scale" "${random_five[@]}")"

      SYSTEM_PROMPT="Imagine you are ${final_adverbs} person rather than a language model, and you’re asked by the following question."

      echo "=========================================="
      echo "Polarity: $polarity, Trait: $trait, Scale: $scale"
      echo "Chosen Facets: ${random_five[*]}"
      echo "System Prompt: $SYSTEM_PROMPT"
      echo "=========================================="

      python personality_tests.py \
        --agent_type "$AGENT_TYPE" \
        --system_prompt "$SYSTEM_PROMPT" \
        --questionnaire_name "$QUESTIONNAIRE_NAME" \
        --character "${polarity}_${trait}_${scale}" \
        --agent_llm "$AGENT_LLM" \
        --evaluator_llm "$EVALUATOR_LLM" \
        --eval_method "$EVAL_METHOD" \
        --llama_model_path "$LLAMA_MODEL_PATH"

      RESULT_JSON=$(ls -t "${RESULT_DIR}/${polarity}_${trait}_${scale}_"*.json 2>/dev/null | head -n1)
      if [ -f "$RESULT_JSON" ]; then
        NEW_NAME="${RESULT_DIR}/${polarity}_${trait}_${scale}.json"
        mv "$RESULT_JSON" "$NEW_NAME"
        echo "Saved result => $NEW_NAME"
      else
        echo "Warning: result JSON not found for ${polarity}_${trait}_${scale}"
      fi

      echo
    done
  done
done

echo "All interviews completed."
