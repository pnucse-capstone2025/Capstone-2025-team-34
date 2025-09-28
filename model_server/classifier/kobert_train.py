import os, random, math
import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score

output_dir = "../_final/classifier/final_kobert_1"

def korean_detector(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        if '\uac00' <= ch <= '\ud7a3':
            return True
    return False

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

def preprocess_kobert(batch):
    enc = kobert_tokenizer(
        batch["question"], batch["article"],
        truncation=True, max_length=512,
        return_token_type_ids=False
    )
    enc.pop("token_type_ids", None)

    enc["labels"] = [type_to_id[t] for t in batch[LABEL_COL]]
    return enc

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

num_train_epochs = 5
per_device_train_batch_size = 2
learning_rate = 2e-5
max_grad_norm = 1.0
warmup_ratio = 0.03

base_paths = "../dataset/분류 모델 용"
data_paths = {
    "cloth": f"{base_paths}/cloth",
    "korean": f"{base_paths}/korean",
    "race_high_long": f"{base_paths}/race_high_long",
    "race_high_short": f"{base_paths}/race_high_short",
    "race_middle_short": f"{base_paths}/race_middle_short",
    "race_middle_long": f"{base_paths}/race_middle_long",
}

nums_list = {
    "cloth": 100,
    "korean": 100,
    "race_high_long": 100,
    "race_high_short": 100,
    "race_middle_short": 100,
    "race_middle_long": 100,
}

type_to_id = {
    "cloth": 0,
    "korean": 1,
    "race_high_long": 2,
    "race_high_short": 3,
    "race_middle_long": 4,
    "race_middle_short": 5,
}
id2label = {i: k for k, i in type_to_id.items()}
label2id = type_to_id
datasets = {}
trains = {}
for key, data_path in data_paths.items():
    print(f"Processing dataset: {key}")
    ds = load_from_disk(data_path)
    datasets[key] = ds
    trains[key] = ds["train"]

train_slices = []
for key, train_ds in trains.items():
    n = min(nums_list.get(key, len(train_ds)), len(train_ds))
    train_slices.append(train_ds.select(range(n)))
train_set = concatenate_datasets(train_slices)

korean_idx = [i for i, ex in enumerate(train_set)]
korean_train_set = train_set.select(korean_idx)

split = korean_train_set.train_test_split(test_size=0.1, seed=SEED, shuffle=True)
kobert_train = split["train"]
kobert_valid = split["test"]
print("Train/Valid sizes:", len(kobert_train), len(kobert_valid))

bad = []
for i in range(len(kobert_train)):
    row = kobert_train[i]
    if not isinstance(row["question"], str) or not isinstance(row["article"], str):
        bad.append((i, type(row["question"]).__name__, type(row["article"]).__name__))

kobert_model_id = "skt/kobert-base-v1"
kobert_tokenizer = AutoTokenizer.from_pretrained(kobert_model_id, use_fast=False)
kobert_model = AutoModelForSequenceClassification.from_pretrained(
    kobert_model_id,
    num_labels=len(type_to_id),
    id2label=id2label,
    label2id=label2id,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kobert_model.to(device)

LABEL_COL = "example_type"

kobert_train_enc = kobert_train.map(preprocess_kobert, batched=True, remove_columns=kobert_train.column_names)
kobert_valid_enc = kobert_valid.map(preprocess_kobert, batched=True, remove_columns=kobert_valid.column_names)
data_collator = DataCollatorWithPadding(tokenizer=kobert_tokenizer)

args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    warmup_ratio=warmup_ratio,
    max_grad_norm=max_grad_norm,
    fp16=False,
    report_to=["none"],
    seed=SEED
)

trainer = Trainer(
    model=kobert_model,
    args=args,
    train_dataset=kobert_train_enc,
    eval_dataset=kobert_valid_enc,
    tokenizer=kobert_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(output_dir)
kobert_tokenizer.save_pretrained(output_dir)
print("KoBERT 저장 경로:", output_dir)