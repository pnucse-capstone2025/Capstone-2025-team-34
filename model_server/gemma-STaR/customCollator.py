import torch
from typing import List, Dict, Any

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template: str):
        self.tokenizer = tokenizer
        self.tmpl_ids: List[int] = tokenizer(response_template, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _find_last_subseq(hay: List[int], needle: List[int]) -> int:
        last = -1
        n = len(needle)
        if n == 0 or len(hay) < n:
            return -1
        for i in range(len(hay) - n + 1):
            if hay[i:i+n] == needle:
                last = i
        return last

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, return_tensors="pt")
        input_ids = batch["input_ids"]
        attn_mask = batch.get("attention_mask", None)
        labels = input_ids.clone()

        tmpl = self.tmpl_ids
        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()
            start = self._find_last_subseq(seq, tmpl)
            if start == -1:
                labels[i, :] = -100
            else:
                cut = start + len(tmpl)
                labels[i, :cut] = -100 

        if attn_mask is not None:
            labels[attn_mask == 0] = -100
        else:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                labels[input_ids == pad_id] = -100

        batch["labels"] = labels
        return batch
