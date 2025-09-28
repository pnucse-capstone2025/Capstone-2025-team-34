import sys
sys.path.append('..')

import torch
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, concatenate_datasets
import os
from datetime import datetime

nums_list = {
    "cloth": 100,
    "korean": 0,
    "race_high_long": 100,
    "race_high_short": 100,
    "race_middle_short": 100,
    "race_middle_long": 100,
}

def korean_detector(text):
    for char in text:
        if '\uac00' <= char <= '\ud7a3':  # 한글 유니코드 범위
            return True
    return False

def half_correct_detector(predicted, actual):
    return predicted[0:4] == actual[0:4] and predicted[-5:] == actual[-5:]

model_id = "microsoft/deberta-v3-base"
save_directory = "../_final/classifier/final_model_deberta"

num_train_epochs=5                
per_device_train_batch_size=2 
learning_rate=2e-5          
max_grad_norm=1.0                    
warmup_ratio=0.03

base_paths = "../_final/dataset/"
data_paths = {
    "cloth": f"{base_paths}/cloth",
    "korean": f"{base_paths}/korean",
    "race_high_long": f"{base_paths}/race_high_long",
    "race_high_short": f"{base_paths}/race_high_short",
    "race_middle_short": f"{base_paths}/race_middle_short",
    "race_middle_long": f"{base_paths}/race_middle_long",
}
num = 500

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=6
)

args = TrainingArguments(
    output_dir="output/results",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    logging_steps=10,
    save_strategy="epoch",
    report_to="tensorboard",
    fp16=True,
)

datasets = {}
trains = {}

for key, data_path in data_paths.items():
    print(f"Processing dataset: {key}")
    dataset = load_from_disk(data_path)
    train = dataset['train']

    datasets[key] = dataset
    trains[key] = train

train_set = []

for key in trains.keys():
    nums = nums_list[key]
    sets = trains[key].select(range(nums))
    sets = sets.filter(lambda example: not korean_detector(example["question"]))
    train_set.append(sets)

train_set = concatenate_datasets(train_set)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type_to_id = {
    "cloth": 0,
    "korean": 1,
    "race_high_long": 2,
    "race_high_short": 3,
    "race_middle_long": 4,
    "race_middle_short": 5,
}

def preprocess_function(examples, tokenizer):
    articles  = list(examples['article'])
    questions = list(examples['question'])

    model_inputs = tokenizer(
        questions,
        articles,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    type_to_id = {"cloth":0,"korean":1,"race_high_long":2,"race_high_short":3,"race_middle_long":4,"race_middle_short":5}
    model_inputs["labels"] = [type_to_id[t] for t in examples["example_type"]]
    return model_inputs

fn_kwargs = {
    "tokenizer": tokenizer,
}

processed_train_set = train_set.map(
    preprocess_function, 
    batched=True, 
    remove_columns=train_set.column_names,
    fn_kwargs=fn_kwargs
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=processed_train_set,
    tokenizer=tokenizer,
)

print("\n모델 학습을 시작합니다...")
trainer.train()
print("모델 학습이 완료되었습니다.")
tests = {}

for key, dataset in datasets.items():
    print(f"Processing dataset: {key}")
    tests[key] = dataset['test']

nums_list = {
    "cloth": 0,
    "korean": 0,
    "race_high_long": 50,
    "race_high_short": 50,
    "race_middle_short": 50,
    "race_middle_long": 50,
}

test_set = []

for key in tests.keys():
    nums = nums_list[key]
    sets = tests[key].select(range(nums))
    test_set.append(sets)

test_set = concatenate_datasets(test_set)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id_to_type = {v: k for k, v in type_to_id.items()}

total = len(test_set)
correct = 0
half_correct = 0

report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = os.path.join(report_dir, f"report_{timestamp}.txt")
report_file = open(report_filename, "w", encoding="utf-8")

for i in range(len(test_set)):
    example = test_set[i]
    
    text = "[article]" + example['article'] + "\\n[question] " + example['question']
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    
    predicted_label = id_to_type[predicted_id]
    actual_label = example['example_type']
    
    report_file.write(f"--- 샘플 {i+1} ---\n")
    report_file.write(f"실제 레이블: {actual_label}\n")
    report_file.write(f"예측 레이블: {predicted_label}\n")
    result_str = '정답' if predicted_label == actual_label else ('부분 정답' if half_correct_detector(predicted_label, actual_label) else '오답')
    report_file.write(f"결과: {result_str}\n")
    report_file.write("-" * 20 + "\n")

    if predicted_label == actual_label:
        correct += 1
    elif half_correct_detector(predicted_label, actual_label):
        half_correct += 1
    else:
        print(f"--- 오답 샘플 {i+1} ---")
        print(f"입력 텍스트: {text[:200]}...")
        print(f"실제 레이블: {actual_label}")
        print(f"예측 레이블: {predicted_label}")
        print("-" * 20)

print(f"전체 샘플 수: {total}")
print(f"정답 수: {correct}")
print(f"정확도: {correct / total * 100:.2f}%")
print(f"정확도(부분 정답 포함): {(correct + half_correct) / total * 100:.2f}%")

report_file.close()

os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"모델과 토크나이저가 '{save_directory}'에 저장되었습니다.")


