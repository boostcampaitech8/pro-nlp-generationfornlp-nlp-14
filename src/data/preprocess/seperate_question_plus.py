# 표준 라이브러리 (Standard Library)
import json
import os

# 서드파티 라이브러리 (Third-party Library)
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils import PreprocessConfig

# 설정 로드 (객체 지향 방식)
config = PreprocessConfig.from_yaml('configs/config.yaml')

# .env 파일에서 환경 변수 로드 (API KEY 등)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def split_paragraph_and_plus(paragraph):
    """
    GPT-4o-mini를 사용해 지문에서 보기 영역을 분리합니다.
    """
    prompt = f"""
    당신은 국어/사회 탐구 지문 분석 전문가 입니다.
    다음 [지문]을 분석하여, 지문 뒤에 붙어 있는 '<보기>'나 '자료' 부분이 있다면 이를 분리해주세요.

    [지문 내용]
    {paragraph}

    [지시 사항]
    1. <보 기>, [자료] 등의 명시적 키워드가 있으면 그 지점부터 분리합니다.
    2. 명시적 키워드가 없더라도, 지문의 설명이 끝나고 갑자기 'ㄱ, ㄴ, ㄷ', 또는 'ⓐ, ⓑ, ⓒ'와 같은 기호가 나열되며 질문에서 인용할 조건을 제시하는 부분은 'question_plus'로 간주합니다.
    3. 분리할 '보기'가 지문에 전혀 포함되어 있지 않다면 'question_plus'는 반드시 'Nan'으로 반환하세요.
    4. 응답은 무조건 아래의 JSON 형식을 따라 응답하세요.

    [응답 형식(JSON)]
    {{
    "paragraph": "정리된 순수 지문",
    "question_plus": "분리된 보기/자료 내용 또는 Nan"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # message -> messages, sysyem -> system 수정
            messages=[{"role": "system", "content": "You are a helpful assistant that outputs JSON"},
                      {"role": "user", "content": prompt}],
            response_format={'type': "json_object"}
        )
        # choice -> choices 수정
        result = response.choices[0].message.content
        # json.load -> json.loads 수정
        data = json.loads(result)

        q_plus = data.get('question_plus', 'Nan')
        # lower() 처리 전 타입 체크 (문자열인 경우만)
        if isinstance(q_plus, str) and q_plus.lower() == 'nan':
            q_plus = np.nan
        return data.get('paragraph', paragraph), q_plus
    except Exception as e:
        print(f"Error: {e}")
        return paragraph, np.nan

# 데이터 로드
df = pd.read_csv(config.input_path)

# tqdm 진행바 활성화
tqdm.pandas()

# apply -> progress_apply로 변경하여 진행 상황 확인
df[['paragraph', 'question_plus']] = df.progress_apply(
    lambda x: split_paragraph_and_plus(x['paragraph']), axis=1, result_type='expand'
)

# 파일 저장
final_output = config.output_path
df.to_csv(final_output, index=False)

print("작업 완료 및 파일 저장 성공!")
