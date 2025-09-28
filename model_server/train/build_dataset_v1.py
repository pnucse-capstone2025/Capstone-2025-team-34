import string
from datasets import load_from_disk, concatenate_datasets
import ast


system_message = (
    "당신은 유능한 영어 문제 풀이 선생님입니다.\n"
    "문제에서 밑줄이라고 한 것은 실제로 지문에서는 강조표시로 나타납니다. **Was Underline**\n"
    "주어진 문제를 읽고 각 선지 별 근거를 작성하고 정답을 마지막에 작성하시오. (선지 하나 당 1줄의 단서, 각 줄 50자 이내, 영어로 작성)\n"
    "예시: "
)

user_prompt = """
[QUESTION]
다음 빈칸에 공통으로 들어갈 말로 가장 적절한 것을 고르시오.

[PASSAGE]
◦ She has a big smile on her _ .
◦ You should learn to _ your problem.

[OPTIONS]
A. face
B. heat
C. meet
D. walk

[ANSWER]
A: Works for smile (noun) and problem (verb “face”).
B: “Heat” is irrelevant, doesn’t fit either sentence.
C: “Meet” fails first use, second also awkward meaning.
D: “Walk” is an action verb, mismatched in both cases.

answer is: A

[QUESTION]
Human beings should care about the environment because _

[PASSAGE]
"Why should I care about the environment?" some people ask...

[OPTIONS]
A. if we make the earth unlivable, we'll have to live in space
B. it can help our children live better
C. if we don't, the earth will become a garbage dump
D. it will help to make the world a good living place

[ANSWER]
A: Space is mentioned, but text says we cannot live in space.
B: Children mentioned, but main point is broader.
C: Garbage dump is example, not full reason.
D: Matches core idea: keep earth suitable for living.

answer is: D

[QUESTION]
According to the passage, _ help children most.

[PASSAGE]
Most people want their children to be successful in school...

[OPTIONS]
A. teachers
B. friends
C. parents
D. classmates

[ANSWER]
A: Teachers important, but cannot handle every child alone.
B: Friends not emphasized as main support.
C: Parents highlighted as crucial for success.
D: Classmates not central to role in text.

answer is: C

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
    dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False, load_from_cache_file=False)
    return dataset

def concat_datasets(datasets_info, segment):
    result = build_dataset("../dataset/my_korean", segment, num=1).select([])

    for dataset_info in datasets_info:
       dataset = build_dataset(dataset_info[0], segment, dataset_info[1])
       result = concatenate_datasets([result, dataset])
    
    return result.shuffle(seed=42)