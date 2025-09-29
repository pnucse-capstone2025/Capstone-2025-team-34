import string
from datasets import load_from_disk, concatenate_datasets
import ast


system_message = (
    "You are a helpful assistant.\n"
    "Provide reasoning only (no mention or hints about options).\n"
    "At the very end, output exactly: 'answer is X' where X is one of A,B,C,D,E.\n"
    "The phrase 'answer is' must appear once, in lowercase, and only on the final line.\n"
    "Do not reveal or imply the answer prior to that final line."
)

user_prompt = """
[QUESTION]
{question}

[PASSAGE]
{article}

[OPTIONS]
{options_block}

[ANSWER]
"""

def format_options(opts):
    letters = string.ascii_uppercase 
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))

def create_conversation(sample):
  opts = sample["options"]
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(article=sample["article"] , question=sample["question"], options_block=format_options(opts))},
      {"role": "assistant", "content": sample["pred"]}
    ]
  }

def create_conversation_for_test(sample):
  opts = sample["options"]
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(article=sample["article"] , question=sample["question"], options_block=format_options(opts))},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }

def build_dataset(data_path, segment, num):
    raw = load_from_disk(data_path)[segment]
    dataset = raw.shuffle(seed=42).select(range(num))
    dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False, load_from_cache_file=False)
    return dataset

def concat_datasets(datasets_info, segment):
    result = build_dataset("../dataset/my_korean", segment, num=1).select([])

    for dataset_info in datasets_info:
       dataset = build_dataset(dataset_info[0], segment, dataset_info[1])
       result = concatenate_datasets([result, dataset])
    return result.shuffle(seed=42)