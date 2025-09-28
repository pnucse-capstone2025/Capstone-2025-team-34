import string
from datasets import load_from_disk, concatenate_datasets
import ast


system_message = (
    "You are a helpful assistant.\n"
    "Given the question text, write a short 3-line strategy in English\n"
    "that explains how to approach solving it."
)

user_prompt = """
Constraints:
- Max 3 lines
- ≤ 25 characters per line
- Directly reference the question’s key words if helpful
- Output only the 3 lines, no extra text

[QUESTION]
The main purpose of the text is to _ .

[OUTPUT]
Focus on the phrase main purpose.
Think about the text’s overall aim.
Match options with global meaning.

[QUESTION]
When the building split into two, the writer and his/her family   _

[OUTPUT]
Notice event: building split.
Think what family would do next.
Pick option matching context.

[QUESTION]
다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?

[OUTPUT]
Focus on underlined phrases.
Check grammar in each phrase.
Find the incorrect structure.

[QUESTION]
{question}

[OUTPUT]
"""

def format_options(opts):
    letters = string.ascii_uppercase 
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))

def create_conversation(sample):
  opts = sample["options"]
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["question"])},
      {"role": "assistant", "content": sample["strategy"]}
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