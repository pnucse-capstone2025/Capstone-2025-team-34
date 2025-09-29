from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 두 모델 모두 로드
use_cpu = True  # CPU 사용 여부 설정
deberta_path = "/home/nclabterm1/refresh/_final/classifier/final_model_deberta"
kobert_path = "/home/nclabterm1/refresh/_final/classifier/final_model_kobert_1"
device = "cuda" if not use_cpu and torch.cuda.is_available() else "cpu"

# DeBERTa 모델 및 토크나이저
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_path)
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to(device).eval()

# KoBERT 모델 및 토크나이저
kobert_tokenizer = AutoTokenizer.from_pretrained(kobert_path, use_fast=False)
kobert_model = AutoModelForSequenceClassification.from_pretrained(kobert_path).to(device).eval()

def korean_detector(text):
    for char in text:
        if '\uac00' <= char <= '\ud7a3':  # 한글 유니코드 범위
            return True
    return False

def route_model(question: str, article: str) -> str:
    return "kobert" if korean_detector(question) else "deberta"

def get_tokenizer_model(route: str):
    if route == "deberta":
        # DeBERTa 모델 및 토크나이저
        return deberta_tokenizer, deberta_model
    elif route == "kobert":
        # KoBERT 모델 및 토크나이저
        return kobert_tokenizer, kobert_model
    return None

def id_to_label(id):
    mapping = ["cloth", "korean", "race_high_long", "race_high_short", "race_middle_long", "race_middle_short"]
    return mapping[id] if 0 <= id < len(mapping) else "unknown"

def build_batch(tokenizer, batch, is_kobert: bool):
    enc = tokenizer(
        batch["question"], batch["article"],
        return_tensors="pt",
        max_length=512, truncation=True, padding=True,
        return_token_type_ids=not is_kobert
    )
    return enc

def run_inference(test_set, models: Dict[str, torch.nn.Module], tokenizers: Dict[str, any], device="cuda"):
    dl = DataLoader(test_set, batch_size=16, shuffle=False)
    id2label = models["deberta"].config.id2label
    stats = dict(total=0, correct=0, half=0, used_k=0, used_d=0, k_correct=0, d_correct=0, d_half=0)

    for batch in dl:
        routes = ["kobert" if korean_detector(q) else "deberta" for q in batch["question"]]
        for route in ["kobert","deberta"]:
            idx = [i for i,r in enumerate(routes) if r==route]
            if not idx: continue
            sub = {k:[v[i] for i in idx] for k,v in batch.items()}
            enc = build_batch(tokenizers[route], sub, is_kobert=(route=="kobert"))
            enc = {k:v.to(device) for k,v in enc.items()}
            with torch.inference_mode():
                logits = models[route](**enc).logits
            pred = logits.argmax(-1).tolist()
            for j, pj in zip(idx, pred):
                predicted = id2label[pj]
                actual = batch["example_type"][j]

def classify(question, article):
    # 한국어 감지하여 모델 선택
    if korean_detector(question):
        model_name = "kobert"
        tokenizer, model = get_tokenizer_model(model_name)
        inputs = tokenizer(
            question, article,
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding="max_length",
            return_token_type_ids=False
        )
    else:
        model_name = "deberta"
        tokenizer, model = get_tokenizer_model(model_name)
        inputs = tokenizer(
            question, article,
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
    
    # 생성된 입력을 모델과 동일한 장치로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 모델을 통해 예측 수행
        outputs = model(**inputs)
    raw_logits = outputs.logits.squeeze().detach().cpu()
    # 로짓을 확률로 변환
    raw_logits = torch.nn.functional.softmax(raw_logits, dim=-1)
    result = {}
    for _ in range(len(raw_logits)):
        result[id_to_label(_)] = float(raw_logits[_])
    return result


if __name__ == "__main__":
    # 테스트용 예시
    question = "빈 칸에 들어갈 말로 알맞은 것은?"
    article = '''Dirty, Unclean, Madness, Estrus
    Libido, Passion, Temptation whoooo
    Dirty, Dirty, Dirty, You think?
    Naughty, Naughty, Naughty, I think,
    Dirty, Unclean, Madness, Estrus
    Libido, Passion, Temptation whoooo
    Wani-Wani-Wani-Warning
    Veri-Veri-Veri-Very
    Fani-Fani-Fani-Funny!
    Yeah, It’s Lunatic time!
    Welcome _ The Madness world!'''
    probs = classify(question, article)
    print("probs:", probs)