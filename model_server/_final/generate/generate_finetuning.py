import re, json, datetime
from tqdm.auto import tqdm
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import os

import prompt.finetune as bd

def safe_letter_token_id(tokenizer, letter):
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
    tokenizer                  
):
    device = next(model.parameters()).device
    model.eval()

    option_letters = ['A', 'B', 'C', 'D', 'E']
    option_token_ids = [safe_letter_token_id(tokenizer, L) for L in option_letters]

    item = bd.create_conversation(item)
    msgs = item['messages']

    prompt_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(prompt_ids)
        next_logits = out.logits[:, -1, :].squeeze(0)
    
    option_logits = next_logits[option_token_ids]   
    option_probs = torch.softmax(option_logits, dim=0)          # (5,)
    probs = option_probs.detach().cpu().float().tolist()

    return probs      
        