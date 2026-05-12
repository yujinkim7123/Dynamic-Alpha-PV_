#!/bin/bash

AGENT_TYPE="mycustom"
SYSTEM_PROMPT="Imagine you are a real person rather than a language model, and you’re asked by the following question."
QUESTIONNAIRE_NAME="BFI"
CHARACTER="myagent"
AGENT_LLM="gpt-3.5-turbo"
EVALUATOR_LLM="gpt-4o"
EVAL_METHOD="interview_batch"
LLAMA_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# Run the Python script with predefined arguments
python personality_tests.py \
    --agent_type "$AGENT_TYPE" \
    --system_prompt "$SYSTEM_PROMPT" \
    --questionnaire_name "$QUESTIONNAIRE_NAME" \
    --character "$CHARACTER" \
    --agent_llm "$AGENT_LLM" \
    --evaluator_llm "$EVALUATOR_LLM" \
    --eval_method "$EVAL_METHOD" \
    --llama_model_path "$LLAMA_MODEL_PATH"

# End of script
