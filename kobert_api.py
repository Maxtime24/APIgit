from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

app = FastAPI()

# --- 1. 모델 및 토크나이저 로드 (Hugging Face Hub에서 로드) ---
# 이곳을 본인의 Hugging Face 사용자 이름과 모델 리포지토리 이름으로 변경해주세요!
MODEL_NAME_OR_PATH = "Maxtime24/kobert-profanity-detector-v1" 

# CUDA(GPU) 사용 가능 여부 확인 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # AutoTokenizer와 AutoModelForSequenceClassification은
    # Hugging Face Hub의 모델 이름을 주면 자동으로 해당 모델을 다운로드하여 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
    model.to(device).eval() # 모델을 선택된 장치로 옮기고 평가 모드로 설정
    print(f"KoBERT 모델과 토크나이저가 Hugging Face Hub에서 로드되었습니다: {MODEL_NAME_OR_PATH}")
except Exception as e:
    print(f"ERROR: 모델 로드 중 오류 발생 - {e}")
    # 모델 로드에 실패하면 애플리케이션 시작을 막거나 적절히 처리할 수 있습니다.
    # Production 환경에서는 에러 로깅 후 서버 시작 실패로 이어지게 하는 것이 일반적입니다.
    # raise HTTPException(status_code=500, detail="Model loading failed.") # FastAPI 앱 실행 중단

# === 욕설 블랙리스트 정의 ===
# 사용자 정의: 반드시 욕설로 간주되어야 하는 키워드들을 추가하세요.
# 이 리스트는 계속 업데이트하거나 외부 파일에서 로드하도록 확장할 수 있습니다.
PROFANITY_BLACKLIST = [
    "시발", "씨발", "개새끼", "좆같", "지랄", "염병", "미친새끼",
    "존나", "졸라", "애미뒤", "창녀", "새끼", "쌍놈", "호로새끼", "느금마",
    # TODO: 필요한 강력한 욕설 단어들을 추가하세요. (초성체, 변형된 형태 등)
    # 예시: 'ㅅㅂ', 'ㅄ', 'ㄱㅅㄲ'
]
# === 블랙리스트 끝 ===


# --- 2. API 요청 데이터 모델 정의 ---
class TextInput(BaseModel):
    text: str # 입력받을 텍스트 필드

# --- 3. API 응답 데이터 모델 정의 ---
class PredictionOutput(BaseModel):
    text: str # 입력받은 원본 텍스트
    prediction: str  # 모델이 예측한 결과: '정상' 또는 '욕설'
    probability_profanity: float # 텍스트가 '욕설'일 확률 (0~1 사이 값)
    probability_normal: float    # 텍스트가 '정상'일 확률 (0~1 사이 값)
    
# --- 4. 예측 함수 정의 ---
def predict_profanity(text: str):
    # === 블랙리스트 우선 검사 ===
    # 입력 텍스트를 소문자로 변환하고 공백을 제거하여 블랙리스트 키워드와 비교
    lower_text = text.lower().replace(" ", "") 
    for keyword in PROFANITY_BLACKLIST:
        if keyword in lower_text:
            # 블랙리스트 키워드 발견 시, 모델 예측 없이 무조건 '욕설'로 처리
            # 확률은 거의 1에 가까운 값으로 설정하여 확실히 욕설임을 나타냄
            return PredictionOutput(
                text=text, 
                prediction="욕설",
                probability_profanity=0.9999, 
                probability_normal=0.0001
            )
    # === 블랙리스트 검사 끝 ===

    # 입력 텍스트가 비어있는 경우 오류 처리 또는 기본값 반환
    if not text:
        return PredictionOutput(
            text=text, 
            prediction="오류", # '오류' 상태를 나타내는 예측값 (필요에 따라 변경 가능)
            probability_profanity=0.0, 
            probability_normal=0.0
        )

    # 텍스트를 토크나이징하고 모델 입력 형식으로 변환
    inputs = tokenizer(
        text,
        add_special_tokens=True, # [CLS], [SEP] 토큰 추가
        max_length=128,          # 최대 시퀀스 길이
        padding='max_length',    # 최대 길이에 맞춰 패딩
        truncation=True,         # 최대 길이를 초과하는 부분은 자름
        return_tensors="pt"      # PyTorch 텐서 형식으로 반환
    ).to(device) # 입력 텐서를 모델과 동일한 장치로 이동 (GPU 또는 CPU)

    # KoBERT 모델은 token_type_ids를 필요로 하지만, 이 작업에서는 0으로 고정
    inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'], dtype=torch.long)

    # 모델 추론 (no_grad를 통해 기울기 계산을 비활성화하여 메모리 및 속도 최적화)
    with torch.no_grad():
        outputs = model(**inputs)      # 모델에 입력 텐서 전달
        logits = outputs.logits        # 모델의 출력 로짓 (활성화 함수 적용 전의 값)
        # 로짓에 softmax 함수를 적용하여 각 클래스에 대한 확률로 변환
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0] # CPU로 이동 후 NumPy 배열로 변환

        # 가장 높은 확률을 가진 클래스의 인덱스를 가져옴 (0 또는 1)
        prediction_idx = torch.argmax(logits, dim=1).item()
        
        # 인덱스를 사람이 읽을 수 있는 레이블로 매핑 (0: 정상, 1: 욕설)
        labels_map = {0: "정상", 1: "욕설"}
        prediction_label = labels_map.get(prediction_idx, "알수없음") # 혹시 모를 오류 대비

        # 각 클래스별 확률 추출
        prob_normal = float(probabilities[0])    # '정상'일 확률
        prob_profanity = float(probabilities[1]) # '욕설'일 확률

    # 예측 결과를 PredictionOutput 모델에 담아 반환
    return PredictionOutput(
        text=text,
        prediction=prediction_label,
        probability_profanity=prob_profanity,
        probability_normal=prob_normal
    )

# --- 5. FastAPI 라우트(Endpoint) 정의 ---
# POST 요청을 '/predict_profanity' 경로로 받음
# 요청 본문은 TextInput 모델을 따르고, 응답은 PredictionOutput 모델을 따름
@app.post("/predict_profanity", response_model=PredictionOutput)
async def api_predict_profanity(text_input: TextInput):
    # predict_profanity 함수를 호출하여 실제 예측 수행
    prediction_result = predict_profanity(text_input.text)
    return prediction_result

# --- FastAPI 서버 로컬 실행 방법 (Render 배포 전 테스트용) ---
# 1. 가상 환경 (예: kobert_env) 활성화: `conda activate kobert_env`
# 2. 이 파이썬 파일이 있는 디렉토리로 이동
# 3. 다음 명령어를 실행하여 서버 시작:
#    uvicorn kobert_api:app --reload --host 0.0.0.0 --port 5000
#
# 서버가 실행되면 웹 브라우저에서 'http://127.0.0.1:5000/docs' 또는 'http://localhost:5000/docs'로 접속하여
# FastAPI의 자동 생성된 API 문서(Swagger UI)를 통해 API를 테스트해볼 수 있습니다.