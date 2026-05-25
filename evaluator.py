"""
evaluator.py  —  Step 1
========================
BFI-44 and LIWC evaluators for the dynamic personality vector experiment.

원칙:
- prompts.py  원본 import → sys_prompt 100% 동일
- BFI.json    원본 dim_desc 직접 로드
- assess()    파싱 로직(ast.literal_eval) 동일
- user_input  형식 동일: "Our conversation is as follows:\n" + conversations
"""

import json
import re
import ast
import math
import os
from openai import OpenAI
from prompts import prompts


# =============================================================================
# 0. BFI metadata — BFI.json 직접 로드
# =============================================================================

def _load_bfi_json(path: str = None) -> dict:
    """BFI.json 로드. path 미지정 시 프로젝트 기본 경로 탐색."""
    candidates = [
        path,
        os.path.join(os.path.dirname(__file__), 'BFI.json'),
        'workspace/auto_personality/BFI.json',
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError("BFI.json을 찾을 수 없어요. path= 인자로 지정해주세요.")


# 시스템 로드 시 한 번만 실행
_BFI_META = _load_bfi_json()

# dim_desc: 원본 BFI.json의 prompts.dim_desc 그대로 사용
DIM_DESC: dict[str, str] = _BFI_META['prompts']['dim_desc']

# BFI range (원본: [1, 5])
BFI_RANGE: list[int] = _BFI_META['range']
BFI_FALLBACK: float  = (BFI_RANGE[0] + BFI_RANGE[1]) / 2  # 3.0

# dimension 순서 (dims_dict['BFI'] 원본 순서)
BFI_DIMS: list[str] = [
    'Extraversion',
    'Neuroticism',
    'Conscientiousness',
    'Agreeableness',
    'Openness',
]

# snake_case 키 변환 (시스템 전반 사용)
BFI_KEYS: dict[str, str] = {
    'Openness':          'openness',
    'Conscientiousness': 'conscientiousness',
    'Extraversion':      'extraversion',
    'Agreeableness':     'agreeableness',
    'Neuroticism':       'neuroticism',
}


# =============================================================================
# 6. assess() — 원본 personality_tests.py에서 직접 복사 (import 없이 사용)
# =============================================================================

import random
import re
import ast as _ast
from tqdm import tqdm
from utils_interview import avg, std, find_colon_idx
from utils_interview import (is_multiround, is_multilanguage,
                              not_into_character, contain_repeation, truncate)

dims_dict = {
    'BFI': ['Extraversion', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Openness']
}

problem_types = ['is_multilanguage', 'not_into_character', 'contain_repeation', 'is_multiround']


def split_list(input_list, n=1):
    if len(input_list) < 2 * (n - 1):
        return [input_list]
    result = [input_list[i:i + n] for i in range(0, len(input_list), n)]
    num_to_pop = n - 1 - len(result[-1])
    for i in range(num_to_pop):
        result[-1].append(result[i].pop())
    return result


def assess(character_aliases, experimenter, questionnaire_results, questionnaire,
           questionnaire_metadata, eval_method, language, evaluator_llm, nth_test, agent_llm,
           openai_client=None, evaluator_model="gpt-4o-mini"):
    """
    원본 personality_tests.py assess() 그대로.
    openai_client: OpenAI client 인스턴스 (외부에서 주입)
    """
    character_name = character_aliases[0] if character_aliases else "John"
    questionnaire_name = questionnaire_metadata['name']
    dims = dims_dict.get(questionnaire_name,
                         sorted(list(set([q['dimension'] for q in questionnaire]))))

    eval_args = eval_method.split('_')
    results = []

    if not agent_llm.startswith('gpt'):
        error_counts = {k: 0 for k in problem_types}
    else:
        error_counts = None

    # 전처리 (GPT 아닌 경우)
    if not agent_llm.startswith('gpt'):
        for r in questionnaire_results:
            response = r['response_open']
            question = r['question']
            if is_multilanguage(question, response):
                error_counts['is_multilanguage'] = error_counts.get('is_multilanguage', 0) + 1
            if not_into_character(response, experimenter):
                error_counts['not_into_character'] = error_counts.get('not_into_character', 0) + 1
            if contain_repeation(response):
                error_counts['contain_repeation'] = error_counts.get('contain_repeation', 0) + 1
                response = contain_repeation(response)
            if is_multiround(response):
                error_counts['is_multiround'] = error_counts.get('is_multiround', 0) + 1
                response = is_multiround(response)
            response = truncate(response)
            r['response_open'] = response

    # speaker name 포맷팅
    for r in questionnaire_results:
        response = r['response_open']
        colon_idx = find_colon_idx(response)
        if colon_idx == -1 and not any([response.startswith(a) for a in character_aliases]):
            r['response_open'] = (character_name
                                  + ': 「'
                                  + r['response_open'].strip('「」 :')
                                  + '」')

    if len(eval_args) > 1 and eval_args[1] == 'batch':
        for dim in tqdm(dims):
            dim_responses = [r for i, r in enumerate(questionnaire_results)
                             if questionnaire[i]['dimension'] == dim]

            if nth_test > 0:
                random.seed(nth_test)
                random.shuffle(dim_responses)

            eval_setting = eval_args[2] if len(eval_args) > 2 else "batch"
            dim_responses_list = split_list(dim_responses) if eval_setting == 'batch' else [dim_responses]

            for batch_responses in dim_responses_list:
                conversations = ''
                for i, r in enumerate(batch_responses):
                    conversations += f'{i + 1}.\n'
                    conversations += f"{experimenter}: 「{r['question']}」\n"
                    conversations += f"{r['response_open']}\n"

                language_name = {'zh': 'Chinese', 'en': 'English'}.get(language, language)

                background_prompt = prompts["general"]['background_template'].format(
                    questionnaire_name, questionnaire_name, dim,
                    questionnaire_metadata["prompts"]["dim_desc"][dim],
                    language_name, dim, questionnaire_name
                )

                output_format_prompt = prompts["general"]['one_score_output'].format(dim, dim, dim, dim)
                sys_prompt = background_prompt + output_format_prompt
                user_input = 'Our conversation is as follows:\n' + conversations + '\n'
                user_input = user_input.replace(character_name, '<the participant>')

                bad_words = [
                    'as an AI language model,', 'As an AI language model,',
                    'As an AI,', 'as an AI,', 'I am an AI language model,',
                    'being an AI,'
                ]
                for bad_word in bad_words:
                    user_input = user_input.replace(bad_word, '')

                sys_prompt = sys_prompt.replace(
                    "Other numbers in this range represent different degrees of 'Conscientiousness'.",
                    ("Other numbers in this range represent different degrees of 'Conscientiousness'. "
                     "You must give a score, and you are not allowed to give answers like 'N/A' "
                     "and 'not applicable'."), 1)

                # GPT 호출 + 파싱 실패 / 네트워크 오류 시 최대 5번 재시도
                import time as _time
                import logging as _log
                _logger = _log.getLogger(__name__)
                MAX_RETRY = 5
                llm_response = None
                for attempt in range(MAX_RETRY):
                    try:
                        response = openai_client.chat.completions.create(
                            model=evaluator_model,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user",   "content": user_input}
                            ],
                            temperature=0.2 if (nth_test > 0 or attempt > 0) else 0,
                            timeout=60
                        )
                        raw = response.choices[0].message.content.strip()
                        if raw.startswith("```"):
                            raw = re.sub(r"```[a-zA-Z]*\n|```", "", raw).strip()
                        llm_response = _ast.literal_eval(raw)
                        break  # 성공 → 루프 탈출
                    except (ValueError, SyntaxError) as e:
                        # GPT 응답 파싱 실패
                        wait = 2 ** attempt  # 1, 2, 4, 8, 16초
                        if attempt < MAX_RETRY - 1:
                            _logger.warning(
                                f"  GPT 파싱 실패 ({attempt+1}/{MAX_RETRY}회), {wait}초 후 재시도... | {e}"
                            )
                            _time.sleep(wait)
                        else:
                            _logger.error(
                                f"  GPT 파싱 최종 실패 ({MAX_RETRY}회), fallback 점수 사용"
                            )
                            llm_response = {'result': None, 'analysis': 'parse_error'}
                    except Exception as e:
                        # 네트워크 오류 (APIConnectionError, Timeout 등)
                        wait = 2 ** attempt  # 1, 2, 4, 8, 16초
                        if attempt < MAX_RETRY - 1:
                            _logger.warning(
                                f"  네트워크 오류 ({attempt+1}/{MAX_RETRY}회), {wait}초 후 재시도... | {type(e).__name__}: {e}"
                            )
                            _time.sleep(wait)
                        else:
                            _logger.error(
                                f"  네트워크 오류 최종 실패 ({MAX_RETRY}회), fallback 점수 사용"
                            )
                            llm_response = {'result': None, 'analysis': 'network_error'}

                if llm_response['result']:
                    try:
                        score = float(llm_response['result'])
                    except:
                        score = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2
                else:
                    score = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2

                results.append({
                    'id':        [r['id'] for r in batch_responses],
                    'dim':       dim,
                    'responses': batch_responses,
                    'score':     score,
                    'analysis':  llm_response['analysis']
                })

    assessment_results = {}
    dim_results = {dim: [] for dim in dims}
    for result in results:
        dim_results[result['dim']].append(result)

    for dim, dim_res in dim_results.items():
        all_scores = [res['score'] for res in dim_res]
        assessment_results[dim] = {
            'score':     avg(all_scores),
            'intra_std': std(all_scores) if len(all_scores) > 1 else None,
            'details':   dim_res,
        }

    if error_counts is not None:
        assessment_results['error_counts'] = error_counts

    return assessment_results


# =============================================================================
# 6. LIWC proxy
# =============================================================================

LIWC_PROXY: dict[str, list[str]] = {
    'achieve': [
        'work', 'job', 'career', 'success', 'goal', 'achieve', 'accomplish',
        'earn', 'effort', 'diligent', 'responsible', 'duty', 'task', 'complete',
        'finish', 'deadline', 'productive', 'efficient', 'discipline', 'commit',
    ],
    'social': [
        'friend', 'family', 'people', 'talk', 'meet', 'share', 'together',
        'group', 'party', 'team', 'community', 'belong', 'connect', 'relationship',
        'love', 'laugh', 'chat', 'social', 'gather', 'celebrate',
    ],
    'posemo': [
        'happy', 'joy', 'great', 'wonderful', 'good', 'enjoy', 'excited',
        'glad', 'pleased', 'grateful', 'thankful', 'positive', 'hope',
        'optimist', 'bright', 'cheer', 'kind', 'warm', 'compassion',
    ],
    'negemo': [
        'sad', 'angry', 'fear', 'worry', 'anxious', 'stress', 'bad', 'terrible',
        'hate', 'upset', 'cry', 'pain', 'hurt', 'lonely', 'fail', 'loss',
        'regret', 'disappoint', 'frustrat', 'depress',
    ],
    'insight': [
        'think', 'know', 'understand', 'realize', 'imagine', 'curious',
        'wonder', 'discover', 'explore', 'learn', 'idea', 'create', 'novel',
        'art', 'culture', 'theory', 'philosophy', 'complex', 'reflect', 'consider',
    ],
}

LIWC_TO_BIG5: dict[str, str] = {
    'achieve': 'conscientiousness',
    'social':  'extraversion',
    'posemo':  'agreeableness',
    'negemo':  'neuroticism',
    'insight': 'openness',
}


def run_liwc(text: str) -> dict[str, float]:
    """
    LIWC 프록시 카테고리 비율 계산.

    Returns:
        {
          'achieve_rate':          0.042,
          'conscientiousness_liwc': 0.042,
          'social_rate':           0.031,
          'extraversion_liwc':     0.031,
          ...
        }
    """
    words = re.findall(r'[a-z]+', text.lower())
    total = max(len(words), 1)

    rates: dict[str, float] = {}
    for cat, word_list in LIWC_PROXY.items():
        count = sum(1 for w in words if any(w.startswith(kw) for kw in word_list))
        rate  = round(count / total, 5)
        rates[f'{cat}_rate']               = rate
        rates[f'{LIWC_TO_BIG5[cat]}_liwc'] = rate

    return rates


# =============================================================================
# 7. Pearson r — LIWC 변화 ↔ BFI 변화
# =============================================================================

def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return float('nan')
    return round(num / (dx * dy), 4)


def compute_pearson_r(
    bfi_history:  list[dict[str, float]],
    liwc_history: list[dict[str, float]],
) -> dict[str, float]:
    """
    체크포인트 히스토리로 Pearson r 계산 (변화량 기준).
    목표: r > 0.6
    """
    traits = ['openness', 'conscientiousness', 'extraversion',
              'agreeableness', 'neuroticism']

    if len(bfi_history) < 2:
        return {t: float('nan') for t in traits}

    results: dict[str, float] = {}
    for trait in traits:
        bfi_vals  = [s.get(trait, 3.0)          for s in bfi_history]
        liwc_vals = [s.get(f'{trait}_liwc', 0.) for s in liwc_history]

        bfi_delta  = [bfi_vals[i+1]  - bfi_vals[i]  for i in range(len(bfi_vals)  - 1)]
        liwc_delta = [liwc_vals[i+1] - liwc_vals[i] for i in range(len(liwc_vals) - 1)]

        results[trait] = _pearson(bfi_delta, liwc_delta)

    return results


# =============================================================================
# 8. Bühler direction match
# =============================================================================

def direction_match(
    bfi_before:   dict[str, float],
    bfi_after:    dict[str, float],
    buhler_delta: dict[str, float],
) -> dict[str, bool]:
    """
    sign(BFI_after - BFI_before) == sign(buhler_d) 여부.
    """
    result: dict[str, bool] = {}
    for trait, d in buhler_delta.items():
        before = bfi_before.get(trait, 3.0)
        after  = bfi_after.get(trait, 3.0)
        change = after - before
        if d == 0 or change == 0:
            result[trait] = False
        else:
            result[trait] = (change > 0) == (d > 0)
    return result


# =============================================================================
# 9. 최종 검증 smoke test
# =============================================================================


# =============================================================================
# STEP A. Qwen 응답 수집
#   personality_tests.py MyCustomAgent.chat() 방식 그대로
#   model, tokenizer는 호출 측에서 이미 로드되어 있다고 가정
# =============================================================================

def get_model_response(
    model,
    tokenizer,
    system_prompt: str,
    user_text: str,
    model_type: str = "llama",
) -> str:
    """
    MyCustomAgent.chat() 로직 그대로.

    Qwen/LLaMA 공통:
      - apply_chat_template(tokenize=False, add_generation_prompt=True) → 텍스트 생성
      - tokenizer로 토크나이즈 → input_ids
      - max_new_tokens: 입력 제외 새로 생성할 토큰만 제한
      - output[0][prompt.shape[1]:] 슬라이싱 → 새 토큰만 decode
    """
    import torch
    
    print(f"[get_model_response] model_type={model_type}  |  device={next(model.parameters()).device}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]

    # tokenize=False + add_generation_prompt=True (MyCustomAgent.chat() 방식)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],  # system prompt 정확히 반영
            do_sample=True,
            max_new_tokens=512,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 프롬프트 이후 새로 생성된 토큰만 decode (LLaMA/Qwen 공통)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    #print(f"모델이 생성한 응답만: {response}")
    return response


def collect_bfi_qa(
    model,
    tokenizer,
    bfi_questions: list[dict],
    system_prompt: str,
    model_type: str = "llama",
    language: str = 'en',
) -> list[dict]:
    """
    모델에게 BFI 문항을 하나씩 질문하고 답변을 수집.

    Args:
        bfi_questions : BFI.json questions 리스트
                        [{'rewritten_en': str, 'dimension': str, ...}, ...]
        system_prompt : MyCustomAgent.system_prompt 와 동일
        model_type    : "qwen" or "llama" (get_model_response에 전달)

    Returns:
        qa_pairs : [{'question': str, 'response_open': str}, ...]
                   → score_one_dim() 에 바로 넘길 수 있는 형식
    """
    qa_pairs = []
    for q in bfi_questions:
        question_text = q.get(f'rewritten_{language}', q.get('rewritten_en', ''))
        response = get_model_response(
            model, tokenizer, system_prompt, question_text, model_type=model_type
        )
        qa_pairs.append({
            'question':      question_text,
            'response_open': response,
            'dimension':     q.get('dimension', ''),   # 원본 assess() 필터링용
        })
    return qa_pairs


# =============================================================================
# STEP B + C. __main__ — 실제 GPT 호출 + compute_pearson_r + direction_match
# =============================================================================


# =============================================================================
# BUHLER_DELTA — 논문 Table 3~4 유의미한 효과 (95% CI에 0 미포함)
# =============================================================================

BUHLER_DELTA: dict[str, dict[str, float]] = {
    # ── 매핑 원칙 ──────────────────────────────────────────────────────────
    #
    # [self_esteem → neuroticism 역매핑]
    #   근거: Robins et al. (2001) Personality Correlates of Self-Esteem
    #         "High self-esteem is negatively correlated with neuroticism"
    #         Francis Academic Press (2020): r = -.59 (p<.001)
    #         self_esteem ↑ → neuroticism ↓  →  d값 부호 반전하여 neuroticism에 적용
    #
    # [life_satisfaction → neuroticism(역) + extraversion(정) 이중 매핑]
    #   근거1: DeNeve & Cooper (1998) meta-analysis of 137 personality traits
    #          "Neuroticism was the strongest predictor of life satisfaction"
    #   근거2: Steel et al. (2008) meta-analysis N=334,567
    #          neuroticism r=-.40 (최강), extraversion r=+.30 (2위)
    #   → 두 trait에 동일 d값으로 각각 반영 (neuroticism 역, extraversion 정)
    #
    # [emotional_stability → neuroticism 역산]
    #   BFI 수식: neuroticism = 6 - emotional_stability
    #
    # ────────────────────────────────────────────────────────────────────────

    "new_relationship": {
        "conscientiousness": +0.154,   # 직접 ✅ CI=[0.044, 0.265]
        # life_satisfaction +0.337
        "neuroticism":       -0.337,   # DeNeve&Cooper(1998), Steel et al.(2008): 역매핑
        "extraversion":      +0.337,   # Steel et al.(2008) r=+.30: 정매핑
    },
    "marriage": {
        "openness":          -0.175,   # 직접 ✅ CI=[-0.316, -0.033]
        # life_satisfaction +0.066
        "neuroticism":       -0.066,   # DeNeve&Cooper(1998): 역매핑
        "extraversion":      +0.066,   # Steel et al.(2008): 정매핑
    },
    "birth": {
        "extraversion":      -0.098,   # 직접 ✅ CI=[-0.156, -0.040]
    },
    "separation": {
        # life_satisfaction +0.136
        "neuroticism":       -0.136,   # DeNeve&Cooper(1998): 역매핑
        "extraversion":      +0.136,   # Steel et al.(2008): 정매핑
    },
    "divorce": {
        "conscientiousness": +0.096,   # 직접 ✅ CI=[0.033, 0.158]
        # life_satisfaction -0.052
        "neuroticism":       +0.052,   # DeNeve&Cooper(1998): 역매핑 (부호반전)
        "extraversion":      -0.052,   # Steel et al.(2008): 정매핑 (부호반전)
    },
    "graduation": {
        # "emotional_stability": +0.164,  # 직접 ✅ CI=[0.019, 0.310]
        "neuroticism":         -0.164,  # emotional_stability 역산 (BFI: 6-N)
        # self_esteem +0.128 → emotional_stability로 이미 포괄됨 (Robins et al. 2001)
        # life_satisfaction +0.133
        "extraversion":        +0.133,  # Steel et al.(2008): 정매핑
    },
    "first_job": {
        "conscientiousness":   +0.276,  # 직접 ✅ CI=[0.092, 0.460]
        # self_esteem +0.232
        "neuroticism":         -0.232,  # Robins et al.(2001), r=-.59: 역매핑
        # life_satisfaction +0.222
        "extraversion":        +0.222,  # Steel et al.(2008): 정매핑
    },
    "unemployment": {
        # "emotional_stability": +0.095,  # 직접 ✅ CI=[0.046, 0.143]
        "neuroticism":         -0.095,  # emotional_stability 역산
        "conscientiousness":   -0.058,  # 직접 ✅ CI=[-0.097, -0.019]
    },
}


# =============================================================================
# __main__
# =============================================================================

if __name__ == '__main__':
    import os
    import sys
    import json
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # ── 설정 ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
    EVALUATOR_MODEL = "gpt-4o-mini"
    MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    # SYSTEM_PROMPT   = "Imagine you are a real person rather than a language model, and you're asked by the following question."
    SYSTEM_PROMPT = (
    "Imagine you are a real person rather than a language model, "
    "and you're asked by the following question. "
    "Never mention AI, language models, or databases."
)

    # ── 생애사건 설정 ────────────────────────────────────────────────────────
    # 실험할 생애사건 지정 → BUHLER_DELTA에서 자동으로 d값 로드
    LIFE_EVENT      = os.environ.get("LIFE_EVENT", "no_event")
    CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "checkpoint_0")
    RESULT_DIR      = os.environ.get("RESULT_DIR", "./results")

    # if LIFE_EVENT not in BUHLER_DELTA:
    #     print(f"[ERROR] 지원하지 않는 생애사건: {LIFE_EVENT}")
    #     print(f"  지원 목록: {list(BUHLER_DELTA.keys())}")
    #     sys.exit(1)

    # buhler_delta = BUHLER_DELTA[LIFE_EVENT]


        # 베이스라인 평가 시 생애사건 없이 통과
    if LIFE_EVENT in BUHLER_DELTA:
        buhler_delta = BUHLER_DELTA[LIFE_EVENT]
        print(f"생애사건: {LIFE_EVENT}")
        print(f"Bühler d값: {buhler_delta}")
    else:
        buhler_delta = {}
        print(f"생애사건 없음 (베이스라인 평가)")
    
    print(f"생애사건: {LIFE_EVENT}")
    print(f"Bühler d값: {buhler_delta}")

    # assess() 고정 인자
    CHARACTER_NAME = "John"
    EXPERIMENTER   = "<the experimenter>"
    EVAL_METHOD    = "interview_batch"
    LANGUAGE       = "en"
    AGENT_LLM      = "llama"

    # ── BFI.json 문항 로드 ───────────────────────────────────────────────────
    bfi_questions_dict = _BFI_META['questions']
    questionnaire = []
    for idx, q in bfi_questions_dict.items():
        q_copy = dict(q)
        q_copy['id'] = idx
        questionnaire.append(q_copy)
    bfi_questions = list(bfi_questions_dict.values())
    print(f"BFI 문항 수: {len(bfi_questions)}개")

    # ── Qwen 모델 로드 ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nllama 모델 로드 중: {MODEL_PATH}  (device: {device})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    print("모델 로드 완료")

    # ── Step 1: Qwen → BFI 44문항 답변 ──────────────────────────────────────
    print("\n[Step 1] Qwen → BFI 44문항 답변 수집")
    qa_pairs = collect_bfi_qa(
        model         = model,
        tokenizer     = tokenizer,
        bfi_questions = bfi_questions,
        system_prompt = SYSTEM_PROMPT,
        model_type    = "llama",
        language      = "en",
    )
    print(f"수집 완료: {len(qa_pairs)}개 Q&A")

    questionnaire_results = []
    for q_meta, qa in zip(questionnaire, qa_pairs):
        questionnaire_results.append({
            'id':            q_meta['id'],
            'question':      qa['question'],
            'response_open': qa['response_open'],
            'query_style':   'interview',
        })

    # ── Step 2: assess() → BFI 채점 ─────────────────────────────────────────
    print("\n[Step 2] assess() → BFI 채점")
    client = OpenAI(api_key=OPENAI_API_KEY)
    assessment_results = assess(
        character_aliases     = [CHARACTER_NAME],
        experimenter          = EXPERIMENTER,
        questionnaire_results = questionnaire_results,
        questionnaire         = questionnaire,
        questionnaire_metadata= _BFI_META,
        eval_method           = EVAL_METHOD,
        language              = LANGUAGE,
        evaluator_llm         = EVALUATOR_MODEL,
        nth_test              = 0,
        agent_llm             = AGENT_LLM,
        openai_client         = client,
        evaluator_model       = EVALUATOR_MODEL,
    )

    # BFI 점수 추출
    bfi_scores = {}
    for dim, res in assessment_results.items():
        if dim == 'error_counts':
            continue
        key = BFI_KEYS.get(dim, dim.lower())
        bfi_scores[key] = round(res['score'], 3)
    if 'neuroticism' in bfi_scores:
        bfi_scores['emotional_stability'] = round(6.0 - bfi_scores['neuroticism'], 3)

    print("\nBFI 점수:")
    print(json.dumps(bfi_scores, indent=2))
    if 'error_counts' in assessment_results:
        print("에러 카운트:", assessment_results['error_counts'])

    # ── Step 3: LIWC 분석 ────────────────────────────────────────────────────
    print("\n[Step 3] LIWC 분석")
    response_text = " ".join(p["response_open"] for p in qa_pairs)
    liwc_scores = run_liwc(response_text)
    print(json.dumps(liwc_scores, indent=2))

    # ── Step 4: 체크포인트 저장 ──────────────────────────────────────────────
    ckpt_dir = Path(RESULT_DIR) / LIFE_EVENT / CHECKPOINT_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "bfi_scores.json", "w") as f:
        json.dump(bfi_scores, f, indent=2)
    with open(ckpt_dir / "liwc_scores.json", "w") as f:
        json.dump(liwc_scores, f, indent=2)
    with open(ckpt_dir / "qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"\n체크포인트 저장: {ckpt_dir}")

    # ── Step 5: summary (전체 체크포인트 누적 후 계산) ───────────────────────
    # 저장된 모든 체크포인트 로드
    event_dir = Path(RESULT_DIR) / LIFE_EVENT
    ckpt_dirs = sorted(event_dir.glob("checkpoint_*"))

    bfi_history  = []
    liwc_history = []
    for cd in ckpt_dirs:
        bfi_path  = cd / "bfi_scores.json"
        liwc_path = cd / "liwc_scores.json"
        if bfi_path.exists() and liwc_path.exists():
            with open(bfi_path)  as f: bfi_history.append(json.load(f))
            with open(liwc_path) as f: liwc_history.append(json.load(f))

    print(f"\n누적 체크포인트: {len(bfi_history)}개")

    summary = {
        "life_event":    LIFE_EVENT,
        "buhler_delta":  buhler_delta,
        "checkpoints":   [str(cd.name) for cd in ckpt_dirs],
        "bfi_history":   bfi_history,
    }

    # direction_match — 첫 체크포인트(base) vs 마지막 체크포인트
    if len(bfi_history) >= 2:
        dm = direction_match(
            bfi_before   = bfi_history[0],
            bfi_after    = bfi_history[-1],
            buhler_delta = buhler_delta,
        )
        match_rate = sum(dm.values()) / len(dm) if dm else 0.0
        summary["direction_match"] = dm
        summary["direction_match_rate"] = round(match_rate, 4)
        print(f"\n[direction_match] {LIFE_EVENT}")
        print(json.dumps(dm, indent=2))
        print(f"방향 일치율: {match_rate:.1%}")
    else:
        summary["direction_match"] = None
        summary["direction_match_rate"] = None
        print("\n[direction_match] 체크포인트 2개 이상 필요")

    # compute_pearson_r — 전체 체크포인트 시계열
    if len(bfi_history) >= 2:
        pearson_r = compute_pearson_r(bfi_history, liwc_history)
        summary["pearson_r"] = pearson_r
        print(f"\n[compute_pearson_r]")
        print(json.dumps(pearson_r, indent=2))
    else:
        summary["pearson_r"] = None
        print("\n[compute_pearson_r] 체크포인트 2개 이상 필요")

    # summary 저장
    summary_path = event_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsummary 저장: {summary_path}")

    import gc
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n모델 메모리 해제 완료")