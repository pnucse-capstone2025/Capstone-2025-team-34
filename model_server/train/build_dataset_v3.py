import string
from datasets import load_from_disk, concatenate_datasets
import ast


system_message = (
    "당신은 유능한 영어 문제 풀이 선생님입니다.\n"
    "문제에서 밑줄이라고 한 것은 실제로 지문에서는 강조표시로 나타납니다. **Was Underline**\n"
    "주어진 지문에서 증거를 찾아 정답을 [ANSWER]밑에 작성하시오."
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
      {"role": "assistant", "content": sample["answer"]}
    ]
  }

def build_dataset(data_path, segment, num):
    raw = load_from_disk(data_path)[segment]
    dataset = raw.shuffle(seed=42).select(range(num))
    dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
    return dataset

def concat_datasets(datasets_info, segment):
    result = build_dataset("../dataset/my_korean", segment, num=1).select([])

    for dataset_info in datasets_info:
       dataset = build_dataset(dataset_info[0], segment, dataset_info[1])
       result = concatenate_datasets([result, dataset])
    
    return result.shuffle(seed=42)