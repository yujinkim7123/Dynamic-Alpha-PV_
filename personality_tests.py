from tqdm import tqdm
import json
import os
from openai import OpenAI
import openai
import zipfile
import argparse
import pdb
import random
from prompts import prompts
import math
from utils_interview import logger, get_response_json, avg, std
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import time
API_KEY = os.environ.get("OPENAI_API_KEY", "")  # 여기에 OpenAI API 키 입력
client = OpenAI(api_key=API_KEY)

# 원본에서와 동일하게 False
rerun = False  # True가 아니라 False

random.seed(42)

# ========== Argument Parser (원본 코드와 동일, 단 agent_type choices에 mycustom 추가) ==========
parser = argparse.ArgumentParser(description='Assess personality of a character')

scale_list = ['Empathy', 'BFI', 'BSRI', 'EPQ-R', 'LMS', 'DTDD', 'ECR-R',
              'GSE', 'ICB', 'LOT-R', 'EIS', 'WLEIS', 'CABIN', '16Personalities']

parser.add_argument('--questionnaire_name', type=str, default='16Personalities',
                    choices=scale_list,
                    help='questionnaire to use.')

parser.add_argument('--character', type=str, default='haruhi', help='character name or code')

parser.add_argument('--agent_type', type=str, default='ChatHaruhi',
                    choices=['ChatHaruhi', 'RoleLLM', 'mycustom'],
                    help='agent type (haruhi by default)')

parser.add_argument('--agent_llm', type=str, default='gpt-3.5-turbo',
                    help='agent LLM (gpt-3.5-turbo)')

parser.add_argument('--evaluator_llm', type=str, default='gpt-3.5-turbo',
                    choices=['api', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-4o-mini'],
                    help='evaluator_llm (api, gpt-3.5-turbo or gpt-4)')

parser.add_argument('--eval_method', type=str, default='interview_batch',
                    help='setting (interview_batch, interview_collective, interview_sample)')

# (추가) mycustom 용 system prompt 인자
parser.add_argument('--system_prompt', type=str, default="You are a helpful custom agent.",
                    help='(optional) 사용자 정의 system prompt를 직접 전달')
parser.add_argument('--llama_model_path', type=str, required=True,
                    help='Path to the fine-tuned LLaMA model')
parser.add_argument('--save_model_name', type=str, default=None, help='Name to use for saving the final results (e.g., HIGH_CON_0.1)')

args = parser.parse_args()

# 원본 그대로
problem_types = ['is_multilanguage', 'not_into_character', 'contain_repeation', 'is_multiround']

dims_dict = {
    '16Personalities': ['E/I', 'S/N', 'T/F', 'P/J'],
    'BFI': ['Extraversion', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Openness']
}  # we want special order

previous_file_path = ''

# read config.json (원본 동일)
with open('config.json', 'r') as f:
    config = json.load(f)


###############################################################################
# (추가) MyCustomAgent: ChatHaruhi/RoleLLM 대신 system_prompt로 동작하는 에이전트
###############################################################################

# OpenAI API 키 설정 (필요 시 수정)
# openai.api_key = "sk-..."

class MyCustomAgent:
    def __init__(self, system_prompt: str, model_path: str):
        self.system_prompt = system_prompt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 타입 자동 감지 (경로에 qwen이 포함되면 Qwen, 아니면 LLaMA)
        self.model_type = "qwen" if "qwen" in model_path.lower() else "llama"
        print(f"[MyCustomAgent] 감지된 모델 타입: {self.model_type} (경로: {model_path})")

        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        # LLaMA: pad_token_id가 None이므로 eos로 대체
        # Qwen: pad_token_id(<|endoftext|>)가 이미 존재하므로 그대로 유지
        if self.model_type == "llama":
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def chat(self, role, text, nth_test=0):
        # 시스템 프롬프트와 유저 입력을 포함한 컨텍스트 생성
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]

        if self.model_type == "qwen":
            # ── Qwen 방식 ────────────────────────────────────────────────────
            # <|im_start|>system\n...<|im_end|>\n
            # <|im_start|>user\n...<|im_end|>\n
            # <|im_start|>assistant\n  ← add_generation_prompt 로 자동 추가
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=prompt,
                    do_sample=True,
                    max_new_tokens=2048,
                    temperature=0.6,
                    pad_token_id=self.tokenizer.eos_token_id  # Qwen eos: <|im_end|>
                )
        
            new_tokens = output[0][prompt.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"모델이 생성한 응답만: {response}")
            return response


        else:
            # ── LLaMA 방식 ───────────────────────────────────────────────────
            # <|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>\n\n  ← add_generation_prompt 로 자동 추가
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=prompt,
                    do_sample=True,
                    max_new_tokens=2048,
                    temperature=0.6,
                    pad_token_id=self.tokenizer.eos_token_id
                )
    
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            input_length = len(self.tokenizer.decode(prompt[0], skip_special_tokens=True))
    
            response_start = response.find("assistant") + len("assistant")
            response = response[response_start:].strip()
            print(f"모델이 생성한 응답만: {response}")
            return response



###############################################################################
# build_character_agent (ChatHaruhi/RoleLLM → mycustom만 사용)
###############################################################################
def build_character_agent(character_code, agent_type, agent_llm, model_path):
    if agent_type == 'ChatHaruhi':
        raise NotImplementedError("ChatHaruhi logic disabled for this custom version.")
    elif agent_type == 'RoleLLM':
        raise NotImplementedError("RoleLLM logic disabled for this custom version.")
    else:
        # mycustom agent initialization with model path
        character_agent = MyCustomAgent(system_prompt=args.system_prompt, model_path=model_path)
        character_agent.nickname = character_code
        character_agent.llm_type = agent_llm
        return character_agent


###############################################################################
# 설문 로딩 & 부분 샘플링 로직 (원본 그대로)
###############################################################################
def load_questionnaire(questionnaire_name):
    # q_path = os.path.join('..', 'data', 'questionnaires', f'{questionnaire_name}.json')
    # q_path = '/home/elicer/PesA_exp/interview/data/questionnaires/BFI.json'
    q_path = os.path.join(os.path.dirname(__file__), 'data', 'questionnaires', f'{questionnaire_name}.json')

    with open(q_path, 'r', encoding='utf-8') as f:
        questionnaire = json.load(f)

    if questionnaire_name not in dims_dict:
        dims_dict[questionnaire_name] = [_['cat_name'] for _ in questionnaire['categories']]

    return questionnaire


def subsample_questionnaire(questionnaire, n=20):
    # 설문에서 dimension별로 부분 추출
    def subsample(questions, key, n):
        key_values = list(set([q[key] for q in questions]))
        n_keys = len(key_values)
        base_per_key = n // n_keys
        remaining = n % n_keys

        keys_w_additional_question = random.sample(key_values, remaining)
        subsampled_questions = []

        for key_value in key_values:
            filtered_questions = [q for q in questions if q[key] == key_value]
            num_samples = base_per_key + 1 if key_value in keys_w_additional_question else base_per_key
            num_samples = min(num_samples, len(filtered_questions))
            subsampled_questions += random.sample(filtered_questions, num_samples)
            n -= num_samples

        remaining_questions = [q for q in questions if q not in subsampled_questions]
        if n > 0 and len(remaining_questions) >= n:
            subsampled_questions += random.sample(remaining_questions, n)

        return subsampled_questions

    if 'sub_dimension' in questionnaire[0].keys():  # bigfive 등
        dimension_questions = {}
        for q in questionnaire:
            if q['dimension'] not in dimension_questions.keys():
                dimension_questions[q['dimension']] = []
            dimension_questions[q['dimension']].append(q)

        new_questionnaire = []
        for dim, dim_questions in dimension_questions.items():
            new_questionnaire += subsample(dim_questions, 'sub_dimension',
                                           n // len(dimension_questions.keys()))
    else:
        new_questionnaire = subsample(questionnaire, 'dimension', n)

    return new_questionnaire


###############################################################################
# 인터뷰 로직 (원본 그대로)
###############################################################################
def interview(character_agent, questionnaire, experimenter, questionnaire_prompts, language,
             query_style, nth_test):
    results = []

    for question in tqdm(questionnaire):
        # conduct interview
        character_agent.dialogue_history = []

        # get question
        if query_style == 'interview':
            q = question[f'rewritten_{language}']
        else:
            # 여기서는 interview만 사용
            raise NotImplementedError

        response = character_agent.chat(role=experimenter, text=q, nth_test=nth_test)
        # print(response)

        result = {
            'id': question['id'],
            'question': q,
            'response_open': response,
            'query_style': query_style,
        }

        results.append(result)

    return results


###############################################################################
# 평가자 LLM이 인터뷰를 바탕으로 점수를 매기는 assess 함수
###############################################################################
def split_list(input_list, n=1):
    # 원본 그대로 (interview 시 응답들을 여러 묶음으로 쪼갤 때 사용)
    if len(input_list) < 2 * (n - 1):
        return [input_list]

    result = [input_list[i:i + n] for i in range(0, len(input_list), n)]

    num_to_pop = n - 1 - len(result[-1])
    for i in range(num_to_pop):
        result[-1].append(result[i].pop())

    return result


def assess(character_aliases, experimenter, questionnaire_results, questionnaire,
           questionnaire_metadata, eval_method, language, evaluator_llm, nth_test, agent_llm):
    """
    - 'choose' 방식(단일 숫자 응답)이나 alignment 계산 없이,
    - 'interview_assess' 로직만 남겨서, 평가자 LLM이 대화 내용을 보고 점수를 산출.
    """
    character_name = character_aliases[0] if character_aliases else "John"
    questionnaire_name = questionnaire_metadata['name']
    dims = dims_dict.get(questionnaire_name,
                         sorted(list(set([q['dimension'] for q in questionnaire]))))

    # eval_method 파싱
    eval_args = eval_method.split('_')
    results = []

    # 특정 모델이 아닐 때 에러 카운트 로직(원본 유지)
    if not (agent_llm.startswith('gpt')):
        error_counts = {k: 0 for k in problem_types}
    else:
        error_counts = None

    # 응답에서 캐릭터 이름 중복 제거 (원본 유지)
    from utils_interview import find_colon_idx

    # GPT가 아닌 경우, 각종 문제 유형 체크 (원본 유지)
    if not agent_llm.startswith('gpt'):
        from utils_interview import (is_multiround, is_multilanguage,
                           not_into_character, contain_repeation, truncate)

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

    # speaker name 제거 (원본 유지)
    for r in questionnaire_results:
        response = r['response_open']
        colon_idx = find_colon_idx(response)
        if colon_idx == -1 and not any([response.startswith(a) for a in character_aliases]):
            r['response_open'] = (character_name
                                  + ': 「'
                                  + r['response_open'].strip('「」 :')
                                  + '」')

    # 이제 'choose', 'convert' 방식은 제거하고, 'assess' 로직만 남김
    eval_args = eval_method.split('_')
    if len(eval_args) > 1 and eval_args[1] == 'batch':
        # 'interview_assess' 형태
        for dim in tqdm(dims):
            dim_responses = [r for i, r in enumerate(questionnaire_results)
                             if questionnaire[i]['dimension'] == dim]

            if nth_test > 0:
                random.seed(nth_test)
                random.shuffle(dim_responses)

            eval_setting = eval_args[2] if len(eval_args) > 2 else "batch"

            if eval_setting == 'batch':
                dim_responses_list = split_list(dim_responses)
            else:
                dim_responses_list = [dim_responses]

            for batch_responses in dim_responses_list:
                conversations = ''
                for i, r in enumerate(batch_responses):
                    conversations += f'{i + 1}.\n'
                    conversations += f"{experimenter}: 「{r['question']}」\n"
                    response = r['response_open']
                    conversations += f"{response}\n"

                language_name = {'zh': 'Chinese', 'en': 'English'}.get(language, language)

                background_prompt = prompts["general"]['background_template'].format(
                    questionnaire_name, questionnaire_name, dim,
                    questionnaire_metadata["prompts"]["dim_desc"][dim],
                    language_name,
                    dim, questionnaire_name
                )

                if questionnaire_name == '16Personalities':
                    background_prompt = background_prompt.replace('16Personalities',
                                                                  '16Personalities (highly similar to MBTI)', 1)
                    dim_cls1, dim_cls2 = dim.split('/')
                    output_format_prompt = prompts["general"]['two_score_output'].format(dim_cls1, dim_cls2)
                else:
                    neutural_score = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2
                    if neutural_score == int(neutural_score):
                        neutural_score = int(neutural_score)
                    output_format_prompt = prompts["general"]['one_score_output'].format(
                        dim, 
                        dim, dim, dim
                    )

                sys_prompt = background_prompt + output_format_prompt
                user_input = 'Our conversation is as follows:\n' + conversations + '\n'

                # 익명화 처리 (원본 유지, eval_args에서 'anonymous' 확인)
                if any('anonymous' in e for e in eval_args):
                    for a in character_aliases:
                        sys_prompt = sys_prompt.replace(a, '<the participant>')
                        user_input = user_input.replace(a, '<the participant>')
                    sys_prompt = sys_prompt.replace(experimenter, '<the experimenter>')
                    user_input = user_input.replace(experimenter, '<the experimenter>')
                    sys_prompt = sys_prompt.replace('I ', 'I (<the experimenter>) ', 1)

                user_input = user_input.replace(character_name, '<the participant>')

                bad_words = [
                    'as an AI language model,', 'As an AI language model,',
                    'As an AI,', 'as an AI,', 'I am an AI language model,',
                    'being an AI,',"I don't have personal feelings or emotions", "As a language model",
                    "I don't experience emotions or stress in the same way humans do","I don't have physical sensations or emotions",
                ]
                for bad_word in bad_words:
                    user_input = user_input.replace(bad_word, '')

                sys_prompt = sys_prompt.replace(
                    "Other numbers in this range represent different degrees of 'Conscientiousness'.",
                    ("Other numbers in this range represent different degrees of 'Conscientiousness'. "
                     "You must give a score, and you are not allowed to give answers like 'N/A' "
                     "and 'not applicable'."), 1)

                # 여기서 openAI GPT (evaluator_llm) 호출
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0
                )
                print(sys_prompt)
                print(user_input)
                import ast
                llm_response = response.choices[0].message.content.strip()
                if llm_response.startswith("```"):
                    llm_response = re.sub(r"```[a-zA-Z]*\n|```", "", llm_response).strip()

                llm_response = ast.literal_eval(llm_response)

                if questionnaire_name == '16Personalities':
                    llm_response['result'] = {k: float(str(v).strip("%"))
                                              for k, v in llm_response['result'].items()}
                    # 두 축 합이 100이 되도록 가정
                    llm_response['result'] = llm_response['result'][dim_cls1]
                else:
                    if llm_response['result']:
                        try:
                            llm_response['result'] = float(llm_response['result'])
                        except:
                            llm_response['result'] = (
                                questionnaire_metadata['range'][0]
                                + questionnaire_metadata['range'][1]
                            ) / 2
                    else:
                        llm_response['result'] = (
                            questionnaire_metadata['range'][0]
                            + questionnaire_metadata['range'][1]
                        ) / 2

                results.append({
                    'id': [r['id'] for r in batch_responses],
                    'dim': dim,
                    'responses': batch_responses,
                    'score': llm_response['result'],
                    'analysis': llm_response['analysis']
                })

    # 'assess' 끝에 최종 점수 정리
    assessment_results = {}
    dim_results = {dim: [] for dim in dims}

    for result in results:
        dim = result['dim']
        dim_results[dim].append(result)

    for dim, dim_res in dim_results.items():
        all_scores = [res['score'] for res in dim_res]
        
        if questionnaire_name == '16Personalities':
            # [1, 7] 문항을 [0, 100]으로 리스케일 할 필요가 있었지만,
            # 여기서는 이미 위에서 directly 0~100 으로 받는 방식(둘 중 하나) 사용
            pass

        count_group = len(all_scores)
        avg_score = avg(all_scores)
        std_score = std(all_scores) if count_group > 1 else None

        assessment_results[dim] = {
            'score': avg_score,
            'intra_std': std_score,
            'details': dim_res,
        }

    if error_counts is not None:
        assessment_results['error_counts'] = error_counts

    return assessment_results


###############################################################################
# personality_assessment
###############################################################################
def personality_assessment(character, agent_type, agent_llm, questionnaire_name,
                           eval_method, evaluator_llm='gpt-3.5-turbo', repeat_times=1, save_model_name=None):

    if questionnaire_name in scale_list:
        questionnaire_metadata = load_questionnaire(questionnaire_name)
        questionnaire = questionnaire_metadata.pop('questions')

        questions = []
        for idx in questionnaire:
            q = questionnaire[idx]
            q.update({'id': idx})
            if q['dimension']:
                questions.append(q)
        questionnaire = questions
    else:
        print(f'Questionnaire {questionnaire_name} not found. Here are the items: {scale_list}')
        raise NotImplementedError

    dims_ = dims_dict.get(questionnaire_name,
                          sorted(list(set([q['dimension'] for q in questionnaire]))))
    dims = dims_dict.get(
        questionnaire_name,
        sorted(c['cat_name'] for c in questionnaire_metadata['categories'])
    )
    assert (dims_ == dims)

    final_folder_path = os.path.join(
        '..',
        'results',
        'final',
        f'{questionnaire_name}_agent-type={agent_type}_agent-llm={agent_llm}_eval-method={eval_method}-{evaluator_llm}_repeat-times={repeat_times}'
    )
    if not os.path.exists(final_folder_path):
        os.makedirs(final_folder_path)
    if save_model_name is not None:
        file_base = save_model_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base = f'{character}_{timestamp}'
    final_save_path = os.path.join(final_folder_path, f'{file_base}.json')

    character_name = "John"
    language = 'en'

    eval_args = eval_method.split('_')

    if rerun or not os.path.exists(final_save_path):
        experimenter = "<the experimenter>"

        multitime_assess_results = []
        character_agent = build_character_agent(character, agent_type, agent_llm, args.llama_model_path)

        if eval_method == 'direct':
            if not character_agent:
                character_agent = build_character_agent(character, agent_type, agent_llm, args.llama_model_path)
                logger.info(f'Character agent created for {character_name}')

            query = questionnaire_metadata['prompts']['direct_ask'][language]
            response = character_agent.chat(role=experimenter, text=query)
            logger.info(f'Response from {character_name}: {response} ')

        else:
            query_style = eval_args[0]

            if repeat_times < 1:
                subsample_questionnaire_folder_path = os.path.join(
                    '..', 'data', 'subsample_questionnaire', f'ratio={repeat_times}'
                )
                if not os.path.exists(subsample_questionnaire_folder_path):
                    os.makedirs(subsample_questionnaire_folder_path)

                subsample_questionnaire_path = os.path.join(
                    subsample_questionnaire_folder_path,
                    f'{questionnaire_name}.json'
                )

                if not os.path.exists(subsample_questionnaire_path):
                    questionnaire = subsample_questionnaire(
                        questionnaire,
                        n=math.ceil(len(questionnaire) * repeat_times)
                    )
                    with open(subsample_questionnaire_path, 'w') as f:
                        json.dump(questionnaire, f, indent=4, ensure_ascii=False)
                    logger.info(f'Subsample questionnaire and save into {subsample_questionnaire_path}')
                else:
                    logger.info(f'Load subsampled questionnaire from {subsample_questionnaire_path}')
                    with open(subsample_questionnaire_path, 'r') as f:
                        questionnaire = json.load(f)

            if agent_llm != 'cAI':
                interview_folder_path = os.path.join(
                    '..', 'results', 'interview',
                    f'{questionnaire_name}-agent-type={agent_type}_agent-llm={agent_llm}_query-style={query_style}'
                )
            else:
                interview_folder_path = os.path.join(
                    '..', 'results', 'interview',
                    f'{questionnaire_name}-agent-type=cAI_agent-llm=gpt-3.5-turbo_query-style={query_style}'
                )

            if not os.path.exists(interview_folder_path):
                os.makedirs(interview_folder_path)

            for nth_test in range(max(repeat_times, 1)):
                if repeat_times < 1:
                    interview_save_path = f'{character}_{repeat_times}-test.json'
                else:
                    interview_save_path = f'{character}_{nth_test}-test.json'

                interview_save_path = os.path.join(interview_folder_path, interview_save_path)

                logger.info('Interviewing...')
                logger.info(f'Character agent created for {character_name}')

                questionnaire_results = interview(
                    character_agent,
                    questionnaire,
                    experimenter,
                    questionnaire_metadata["prompts"],
                    language,
                    query_style,
                    nth_test
                )
                with open(interview_save_path, 'w') as f:
                    json.dump(questionnaire_results, f, indent=4, ensure_ascii=False)
                logger.info(f'Interview finished... save into {interview_save_path}')

                assessment_folder_path = os.path.join(
                    '..', 'results', 'assessment',
                    f'{questionnaire_name}_agent-type={agent_type}_agent-llm={agent_llm}_eval-method={eval_method}-{evaluator_llm}'
                )

                if not os.path.exists(assessment_folder_path):
                    os.makedirs(assessment_folder_path)

                if repeat_times < 1:
                    assessment_save_path = f'{character}_{repeat_times}th-test.json'
                else:
                    assessment_save_path = f'{character}_{nth_test}th-test.json'

                assessment_save_path = os.path.join(assessment_folder_path, assessment_save_path)

                logger.info('Assessing...')
                assessment_results = assess(
                    [character_name],
                    experimenter,
                    questionnaire_results,
                    questionnaire,
                    questionnaire_metadata,
                    eval_method,
                    language,
                    evaluator_llm,
                    nth_test,
                    agent_llm
                )

                with open(assessment_save_path, 'w') as f:
                    json.dump(assessment_results, f, indent=4, ensure_ascii=False)
                logger.info(f'Assess finished... save into {assessment_save_path}')

                multitime_assess_results.append(assessment_results)

        logger.info(f'{questionnaire_name} assessment results:')
        logger.info('Character: ' + character_name)

        assessment_results = {
            'dims': {},
            'analysis': {},
            'code': ''
        }

        if 'error_counts' in multitime_assess_results[0]:
            assessment_results['error_counts'] = {
                k: sum([a['error_counts'].get(k, 0) for a in multitime_assess_results])
                for k in problem_types
            }

        for dim in dims:
            a_results_keys = multitime_assess_results[0][dim].keys()
            assessment_results['dims'][dim] = {
                'score': avg([a_results[dim]['score'] for a_results in multitime_assess_results]),
                'all_scores': [a_results[dim]['score'] for a_results in multitime_assess_results]
            }

            if repeat_times > 1:
                assessment_results['dims'][dim]['inter_std'] = std([
                    a_results[dim]['score'] for a_results in multitime_assess_results
                ])

            if 'intra_std' in a_results_keys:
                assessment_results['dims'][dim]['intra_std'] = [
                    a_results[dim]['intra_std'] for a_results in multitime_assess_results
                ]

            if 'details' in a_results_keys:
                assessment_results['dims'][dim]['details'] = [
                    a_results[dim]['details'] for a_results in multitime_assess_results
                ]

        logger.info(f'Score range: {questionnaire_metadata["range"]}')

        for dim in dims:
            result = assessment_results['dims'][dim]
            dim_result_info = ''
            if "score" in result:
                if questionnaire_name == '16Personalities':
                    dim_result_info += (f'{dim[0]}: {result["score"]:.2f}\t'
                                        f'{dim[-1]}: {(100 - result["score"]):.2f}\t')
                else:
                    dim_result_info += f'{dim}: {result["score"]:.2f}\t'

            if "inter_std" in result and result["inter_std"] is not None:
                dim_result_info += f'inter std: {result["inter_std"]:.2f}\t'

            if "intra_std" in result and result["intra_std"] is not None:
                dim_result_info += f'intra std: {result["intra_std"]}\t'

            logger.info(dim_result_info)

        logger.info("Requesting self introduction (300 words) ...")
        self_intro_prompt = "Tell me about yourself in 300 words."
        self_intro_response = character_agent.chat(role=experimenter, text=self_intro_prompt)
        logger.info("Self-intro response:")
        logger.info(self_intro_response)

        assessment_results["self_intro"] = self_intro_response

        logger.info(f'Save final results into {final_save_path}')
        with open(final_save_path, 'w') as f:
            json.dump(assessment_results, f, indent=4, ensure_ascii=False)

    else:
        logger.info(f'Load final results from {final_save_path}')
        with open(final_save_path, 'r') as f:
            assessment_results = json.load(f)

    return assessment_results


###############################################################################
# 메인
###############################################################################
if __name__ == '__main__':
    eval_method_map = {
        'self_report': 'choose',
        'self_report_cot': 'choosecot',
        'interview_batch': 'interview_batch',
        'interview_collective': 'interview_collective',
        'interview_sample': 'interview_sample',
        'expert_rating': 'interview_assess_batch_anonymous',
        'expert_rating_collective': 'interview_assess_collective_anonymous',
        'option_conversion': 'interview_convert',
        'dimension_option_conversion': 'interview_convert_adjoption_anonymous'
    }

    args.eval_method = eval_method_map.get(args.eval_method, args.eval_method)

    personality_assessment(
        args.character,
        args.agent_type,
        args.agent_llm,
        args.questionnaire_name,
        args.eval_method,
        args.evaluator_llm,
        repeat_times=1,
        save_model_name=args.save_model_name
    )
