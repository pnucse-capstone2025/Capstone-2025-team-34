import glob
import json
import math, numpy as np
from typing import Dict, List, Tuple, Optional

ETA = 12.0 # 정확도 -> 가중치 변환 계수
ACC_EPS = 1e-3 # 정확도 클리핑
DROP_BELOW_CHANCE = True
TAU_MODEL = 1.0 # 모델 로짓 온도

base_path = "../../"
model_paths = {
    "gemma-finetuning" : base_path + "gemma-finetuning",
    "gemma-STaR" : base_path + "gemma-STaR",
    "gemma-teacher-student" : base_path + "gemma-teacher-student",
    "phi-finetuning" : base_path + "phi-finetuning",
    "phi-STaR" : base_path + "phi-STaR",
    "phi-teacher-student" : base_path + "phi-teacher-student",
}
eval_path = "/eval_2/"

model_accuracies_by_label = {
    "cloth": {},
    "korean": {},
    "race_high_long": {},
    "race_high_short": {},
    "race_middle_long": {},
    "race_middle_short": {}
}

for model_name, path in model_paths.items():
    target = path + eval_path
    patterns = {
        "cloth": "results_ds_cloth_*.json",
        "korean": "results_ds_korean_*.json",
        "race_high_long": "results_ds_race_high_long_*.json",
        "race_high_short": "results_ds_race_high_short_*.json",
        "race_middle_long": "results_ds_race_middle_long_*.json",
        "race_middle_short": "results_ds_race_middle_short_*.json"
    }
    for label, pattern in patterns.items():
        file = glob.glob(target + pattern)
        accuracy = None
        with open(file[0], 'r') as f:
            data = json.load(f)
            if "accuracy_no_skipped" in data:
                accuracy = data["accuracy_no_skipped"]
        if accuracy is not None:
            model_accuracies_by_label[label][model_name] = accuracy

def _as1d_float_array(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)
    elif a.ndim > 1:
        a = a.reshape(-1)
    return a

def _clean_model_logits_dict(d):
    out = {}
    for k, v in d.items():
        a = _as1d_float_array(v)
        if a.size == 0 or not np.isfinite(a).all():   # 빈 배열/NaN/inf 제거
            continue
        out[k] = a
    return out

def _clean_type_logits_dict(d):
    out = {}
    for k, v in d.items():
        a = _as1d_float_array(v)
        if a.size == 0 or not np.isfinite(a[0]):
            continue
        out[k] = float(a[0])
    return out

def _softmax(z: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(tau, 1e-8)
    z -= z.max()
    ez = np.exp(z)
    return ez / (ez.sum() + 1e-12)

def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))

def _to_float01(x) -> float:
    # 입력이 숫자/문자열/퍼센트 문자열인 경우 [0,1] 실수로 변환
    if isinstance(x, (int, float)):
        return float(x) if x <= 1.0 else float(x) / 100.0
    s = str(x).strip().rstrip('%')
    return float(s) / 100.0 if '%' in str(x) else float(s)

def _chance_base(K: int) -> float:
    return 1.0 / max(K, 1)

def _build_weights_soft_type(acc_table: Dict[str, Dict[str, float]],
                             type_probs: Dict[str, float],
                             available_ids: List[str],
                             K: int) -> Dict[str, float]:
    if not available_ids:
        return {}
    # 1) 유형 posterior
    type_names = list(type_probs.keys())
    q_vals = np.array([type_probs[t] for t in type_names], dtype=np.float64)
    q_sum = q_vals.sum()
    if q_sum > 1e-8:
        q_vals /= q_sum
    else: # 합이 0에 가까우면 균등 분배
        q_vals = np.ones(len(type_names), dtype=np.float64) / len(type_names)
    q = q_vals
    # 2) 모델 가중치 합성
    base = _chance_base(K)
    raw = np.zeros(len(available_ids), dtype=np.float64)
    for ti, t in enumerate(type_names):
        row = acc_table.get(t, {})
        if not row:
            continue
        for mi, mid in enumerate(available_ids):
            a = row.get(mid)
            if a is None:
                print(f"Warning: accuracy missing for type '{t}', model '{mid}'")
                continue
            a = max(min(_to_float01(a), 1.0 - ACC_EPS), ACC_EPS)
            if DROP_BELOW_CHANCE and a < base:
                contrib = 0.0
            else:
                contrib = math.exp(ETA * (a - base))
            raw[mi] += q[ti] * contrib
    # 3) 정규화
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        if not np.isfinite(raw).all():
            print("Warning: non-finite weights:", raw)
        elif raw.sum() <= 0:
            print("Warning: zero-sum weights:", raw)
        return {mid: 1.0 / len(available_ids) for mid in available_ids}
    raw /= raw.sum()
    return {mid: float(w) for mid, w in zip(available_ids, raw)}

def weighted_output(model_probs: Dict[str, np.ndarray], type_probs: Dict[str, float]) -> Dict[str, float]:
    # 입력 강제 변환
    model_probs = _clean_model_logits_dict(model_probs)
    type_probs  = _clean_type_logits_dict(type_probs)

    # 클래스 수 K 정하기(가용 모델 중 첫 번째 길이)
    K = int(next(iter(model_probs.values())).shape[0])
    # K가 다른 모델은 자동 제외
    model_probs = {m: v for m, v in model_probs.items() if v.shape[0] == K}
    if not model_probs:
        raise ValueError("model_probs가 비었습니다.")
    first = next(iter(model_probs.values()))
    K = int(np.asarray(first).shape[-1])
    letters = [chr(ord('A') + j) for j in range(K)]

    available = [m for m, v in model_probs.items() if np.asarray(v).shape[-1] == K]
    if not available:
        raise ValueError("선택지 수 K와 맞는 모델 확률이 없습니다.")

    W = _build_weights_soft_type(model_accuracies_by_label, type_probs, available, K)

    logp = np.zeros(K, dtype=np.float64)
    for mid, w in W.items():
        p = np.asarray(model_probs[mid], dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        logp += w * np.log(p)
    ens_probs = np.exp(logp - _logsumexp(logp))

    pred_idx = int(np.argmax(ens_probs))
    pred_letter = letters[pred_idx]

    type_names = list(type_probs.keys())
    q_vals = np.array([type_probs.get(t, 0.0) for t in type_names], dtype=np.float64)
    q_sum = q_vals.sum()
    if q_sum > 1e-8:
        q_vals /= q_sum
    else:
        q_vals = np.ones(len(type_names), dtype=np.float64) / max(1, len(type_names))
    type_post = {t: float(qi) for t, qi in zip(type_names, q_vals)}
    probs_list = ens_probs.tolist()

    return pred_letter, probs_list, W, type_post

if __name__ == "__main__": # 테스트용
    # 예시 데이터
    model_probs = {
        'gemma-finetuning': [0.1982421875, 0.306640625, 0.10595703125, 0.099609375, 0.2890625],
        'gemma-STaR': [0.06298828125, 0.33984375, 0.30078125, 0.09130859375, 0.2060546875],
        'gemma-teacher-student': [0.296875, 0.140625, 0.12353515625, 0.3359375, 0.1025390625],
        'phi-finetuning': [0.0260009765625, 0.5234375, 0.2470703125, 0.05517578125, 0.1494140625],
        'phi-STaR': [0.240234375, 0.65234375, 0.019775390625, 0.0537109375, 0.032470703125], 
        'phi-teacher-student': [1.0, 1.895427703857422e-05, 1.895427703857422e-05, 1.0132789611816406e-05, 2.1457672119140625e-05]
    }
    type_probs = {
        "korean": 0.6, "cloth": 0.05,
        "race_high_long": 0.15, "race_high_short": 0.05,
        "race_middle_long": 0.1, "race_middle_short": 0.05
    }
    print("--- 레이블별 모델 정확도 ---")
    print(json.dumps(model_accuracies_by_label, indent=2))
    print("---------------------------------")

    pred, probs, weights, type_post = weighted_output(model_probs, type_probs)
    print("pred:", pred)
    print("weights:", weights)
    print("type posterior:", type_post)
    print("probs:", probs)