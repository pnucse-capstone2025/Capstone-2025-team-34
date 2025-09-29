import re, json, datetime
from tqdm.auto import tqdm
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import prompt.distill as bd

class StopOnAnswerIs(StoppingCriteria):
    def __init__(self, tokenizer, pattern=r"answer\s*is\s*:?\s*$", lookback_tokens=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.regex = re.compile(pattern, flags=re.IGNORECASE)
        self.lookback = lookback_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        tail = input_ids[0, -self.lookback:].tolist()
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        return bool(self.regex.search(text))

def safe_letter_token_id(tokenizer, letter: str) -> int:
    vocab = tokenizer.get_vocab()
    letter = letter.upper()
    for tok in (f"▁{letter}", f" {letter}", letter):
        if tok in vocab:
            return vocab[tok]
    ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    raise ValueError(f"'{letter}'가 단일 토큰이 아닙니다. (ids={ids})")

def evaluate_answer(
    item,                   
    model,
    tokenizer,
    do_sample=False,          # 생성 전략(그리디/샘플링)
    temperature=0.7,
    top_p=0.9,
    lookback_tokens=64,
):
    device = next(model.parameters()).device
    model.eval()

    stopper = StoppingCriteriaList([StopOnAnswerIs(tokenizer, lookback_tokens=lookback_tokens)])
    option_letters = ['A', 'B', 'C', 'D', 'E']
    option_token_ids = [safe_letter_token_id(tokenizer, L) for L in option_letters]
    item = bd.create_conversation_for_test(item)
    msgs = item['messages']

    # 1) 프롬프트 구성
    prompt_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    # 2) 생성 시작 → 'answer is'에서 정지
    gen_out = model.generate(
        prompt_ids,
        max_new_tokens=256,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        stopping_criteria=stopper,
        return_dict_in_generate=True,
        use_cache=True,
    )
    seq = gen_out.sequences
    gen_text = tokenizer.decode(seq[0, prompt_ids.shape[1]:], skip_special_tokens=True)

    if not re.search(r"answer\s*is\s*:?\s*$", gen_text.strip(), re.IGNORECASE):
        return None

    with torch.inference_mode():
        out2 = model(seq.to(device))
        next_logits = out2.logits[:, -1, :].squeeze(0)  

    option_logits = next_logits[option_token_ids]   
    option_probs = torch.softmax(option_logits, dim=0)          # (5,)
    probs = option_probs.detach().cpu().float().tolist()

    return probs       
        