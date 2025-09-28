import sys
import os

import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
sys.path.append("../")
import generate.generate_finetuning
import generate.generate_STaR
import generate.generate_teacher_student
import classifier.classifier
import classifier.weighted
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
login(token)

def pick_eval_fn(adapter_name: str):
    if adapter_name.endswith("finetuning"):
        return generate.generate_finetuning.evaluate_answer
    if adapter_name.endswith("STaR"):
        return generate.generate_STaR.evaluate_answer
    if adapter_name.endswith("teacher-student"):
        return generate.generate_teacher_student.evaluate_answer

def pick_base(adapter_name: str):
    if adapter_name.startswith("gemma-"):
        return gemma, gemma_tokenizer
    if adapter_name.startswith("phi-"):
        return phi, phi_tokenizer

model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
    )

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

gemma_base = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it",**model_kwargs,)
gemma = PeftModel.from_pretrained(
    gemma_base, "../model/gemma-finetuning",
    adapter_name="gemma-finetuning", is_trainable=False
)
gemma.load_adapter("../model/gemma-STaR", adapter_name="gemma-STaR")
gemma.load_adapter("../model/gemma-teacher-student", adapter_name="gemma-teacher-student")
gemma.eval()
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)

phi_base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct",**model_kwargs,)
phi = PeftModel.from_pretrained(
    phi_base, "../model/phi-finetuning",
    adapter_name="phi-finetuning", is_trainable=False
)
phi.load_adapter("../model/phi-STaR", adapter_name="phi-STaR")
phi.load_adapter("../model/phi-teacher-student", adapter_name="phi-teacher-student")
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)

ADAPTER_NAMES = [
    "gemma-finetuning",
    "gemma-STaR",
    "gemma-teacher-student",
    "phi-finetuning",
    "phi-STaR",
    "phi-teacher-student",
]

app = FastAPI()

@app.post("/mcq")
async def mcq(req: Request):
    body = await req.json()
    article = body["article"]
    question = body["question"]
    options = body["options"]

    results = {}
    item = body

    classified = classifier.classifier.classify(question, article)

    print(classified)

    for adapter_name in ADAPTER_NAMES:
        eval_fn = pick_eval_fn(adapter_name)
        model, tok = pick_base(adapter_name)
        model.set_adapter(adapter_name)

        option_logits = await run_in_threadpool(eval_fn, item, model, tok)

        results[adapter_name] = option_logits
        print(adapter_name + " done.")

        del model
        torch.cuda.empty_cache()
    
    for k, v in list(results.items()):
        if v is None:
            del results[k]
    print(results)
    
    pred_letter, ens_probs, W, type_post = classifier.weighted.weighted_output(results, classified)

    return JSONResponse({"answer": pred_letter, "final_probs": ens_probs, "classified":classified, "model_probs":results, "weight":W})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)