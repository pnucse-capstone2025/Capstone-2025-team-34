import datetime
import os
import sys
sys.path.append('../..')

import classifier
import weighted
import glob
import json
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from tqdm import tqdm

def probs_to_logits_softmax(p, tau=1.0, eps=1e-12):
    """멀티클래스 확률 -> softmax용 logits (동치)."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if not np.isclose(s, 1.0): p = p / (s + eps)
    p = np.clip(p, eps, 1.0 - eps)
    z = tau * np.log(p)
    return z

base_path = "../../"
model_paths = {
    "gemma-finetuning" : base_path + "gemma-finetuning",
    "gemma-STaR" : base_path + "gemma-STaR",
    "gemma-teacher-student" : base_path + "gemma-teacher-student",
    "phi-finetuning" : base_path + "phi-finetuning",
    "phi-STaR" : base_path + "phi-STaR",
    "phi-teacher-student" : base_path + "phi-teacher-student",
}
eval_path = "/eval_1/"
dataset_base_path = "../dataset/"
dataset_paths = {
    "cloth": dataset_base_path + "cloth",
    "korean": dataset_base_path + "korean",
    "race_high_long": dataset_base_path + "race_high_long",
    "race_high_short": dataset_base_path + "race_high_short",
    "race_middle_long": dataset_base_path + "race_middle_long",
    "race_middle_short": dataset_base_path + "race_middle_short"
}

datasets = {}
tests = {}

for key, data_path in dataset_paths.items():
    print(f"Processing dataset: {key}")
    dataset = load_from_disk(data_path)
    datasets[key] = dataset
    tests[key] = dataset['test']


model_results_by_label = {
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
        results = None
        with open(file[0], 'r') as f:
            data = json.load(f)
            if "results" in data:
                results = data["results"]
        if results is not None:
            model_results_by_label[label][model_name] = results

model_results_weighted_by_label = {
    "cloth": {},
    "korean": {},
    "race_high_long": {},
    "race_high_short": {},
    "race_middle_long": {},
    "race_middle_short": {}
}

numbers_of_samples = {
    "korean": 62,
    "cloth": 238,
    "race_middle_long": 87,
    "race_middle_short": 87,
    "race_high_long": 262,
    "race_high_short": 262
}

for label, num_samples in numbers_of_samples.items():
    for i in range(num_samples):
        sample_data = {}
        sample_data['question'] = tests[label][i]['question']
        sample_data['article'] = tests[label][i]['article']
        sample_data['answer'] = tests[label][i]['answer']
        sample_data['probs_by_model'] = {}
        for model_name, results in model_results_by_label[label].items():
            probs = results[i]['probs']
            sample_data['probs_by_model'][model_name] = probs
        model_results_weighted_by_label[label][i] = sample_data

results_data = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_results": {}
}

report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = os.path.join(report_dir, f"report_mix_{timestamp}.json")


results_data = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

total_correct = 0
total_samples = 0

dataset_results = {}

for label, samples in model_results_weighted_by_label.items():
    correct = 0
    total = len(samples)
    print(f"\n처리 중인 데이터셋: {label}")

    dataset_samples_results = []
    for i, sample in tqdm(samples.items(), total=total, desc=f"{label} 처리 중", ncols=100):
        question = sample['question']
        article = sample['article']
        classified = classifier.classify(question, article)
        model_probs = {}
        for model_name, probs in sample['probs_by_model'].items():
            probs_list = [x for x in probs.values()]
            model_probs[model_name] = probs_list
        pred, probs, weights, type_post = weighted.weighted_output(model_probs, classified)
        sample_result = {
            "id": i,
            "question": question,
            "article": article[:100] + "..." if len(article) > 100 else article,
            "prediction": pred,
            "answer": sample['answer'],
            "correct": pred == sample['answer'],
            "classification": classified,
            "weights": weights,
            "type_post": type_post
        }
        dataset_samples_results.append(sample_result)
        if pred == sample['answer']:
            correct += 1
    
    total_correct += correct
    total_samples += total

    accuracy = correct / total if total > 0 else 0.0
    print(f"{label} 정확도: {accuracy:.4f} ({correct}/{total})")

    dataset_results[label] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "samples": dataset_samples_results
    }

overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
print(f"\n전체 정확도: {overall_accuracy:.4f} ({total_correct}/{total_samples})")

results_data["overall"] = {
    "accuracy": overall_accuracy,
    "correct": total_correct,
    "total": total_samples
}

results_data["dataset_results"] = dataset_results

with open(report_filename, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

print(f"\n결과가 {report_filename}에 저장되었습니다.")