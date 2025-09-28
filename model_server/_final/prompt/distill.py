import string

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

def create_conversation_for_test(sample):
  opts = sample["options"]
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(article=sample["article"] , question=sample["question"], options_block=format_options(opts))}
    ]
  }