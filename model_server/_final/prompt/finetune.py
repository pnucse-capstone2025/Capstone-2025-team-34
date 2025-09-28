import string

system_message = (
    "You are a helpful AI assistant."
    "Please answer the user's questions correctly."
    "Look for the evidence in the text when answering."
    "Underlined replaced with highlights. Example: **Was Underline**"
    "Return only the letter corresponding to the best answer."
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
    ]
  }