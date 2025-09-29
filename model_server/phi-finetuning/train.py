from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import build_dataset as bd 
from pathlib import Path
import json
from collections import OrderedDict
import os
import sys
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

model_id = "microsoft/Phi-4-mini-instruct"

torch_dtype = torch.bfloat16

lora_alpha = 32
lora_dropout=0.05
r=16

num_train_epochs=3
per_device_train_batch_size=4
per_device_eval_batch_size=4
learning_rate=2e-4
max_grad_norm=1.0
warmup_ratio=0.03

use_train_datasets_list = [
    ["../dataset/my_korean", 1], 
    ["../dataset/my_race_middle", 1], 
    ["../dataset/my_race_high", 1], 
    ["../dataset/my_cloth", 1]
    ]

use_validation_datasets_list = [
    ["../dataset/my_korean", 1], 
    ["../dataset/my_race_middle", 1],
    ["../dataset/my_race_high", 1],
    ["../dataset/my_cloth", 1],
    ]


def next_number_str(str_list: list[str]) -> str:
    if len(str_list) == 0:
        return "0"
    return str(int(max(str_list, key=int)) + 1)

def main():
    login(token)
    os.makedirs('./output', exist_ok=True)
    corrent_model_list = os.listdir('./output')
    next_model_name = next_number_str(corrent_model_list)
    output_dir = "output/"+next_model_name

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch_dtype, 
        device_map="auto", 
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save=["lm_head", "embed_tokens"]
    )

    args = SFTConfig(
        output_dir=output_dir, 
        max_length=512,           
        packing=True,             
        gradient_accumulation_steps=4,    
        gradient_checkpointing=True,       
        optim="adamw_torch_fused",      
        logging_steps=10,                   
        save_strategy="epoch",
        fp16=True if torch_dtype == torch.float16 else False,  
        bf16=True if torch_dtype == torch.bfloat16 else False, 

        eval_strategy="epoch",
        per_device_eval_batch_size=per_device_eval_batch_size,

        num_train_epochs=num_train_epochs,                
        per_device_train_batch_size=per_device_train_batch_size,  
        learning_rate=learning_rate,           
        max_grad_norm=max_grad_norm,                    
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="constant",         
        push_to_hub=False,                     
        report_to="tensorboard",         
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
            }
    )

    info_data = OrderedDict()
    info_data["model_id"] = model_id
    info_data["torch_dtype"] = torch_dtype
    info_data["lora_alpha"] = lora_alpha
    info_data["lora_dropout"] = lora_dropout
    info_data["r"] = r
    info_data["num_train_epochs"] = num_train_epochs
    info_data["per_device_train_batch_size"] = per_device_train_batch_size
    info_data["per_device_eval_batch_size"] = per_device_eval_batch_size
    info_data["learning_rate"] = learning_rate
    info_data["max_grad_norm"] = max_grad_norm
    info_data["warmup_ratio"] = warmup_ratio
    info_data["use_train_datasets_list"] = use_train_datasets_list
    info_data["use_validation_datasets_list"] = use_validation_datasets_list

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_bos_token = False

    train_dataset = bd.concat_datasets(use_train_datasets_list, "train")
    eval_dataset = bd.concat_datasets(use_validation_datasets_list, "validation")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl)) 

    ids     = batch["input_ids"][0].tolist()
    labels  = batch["labels"][0].tolist()
    tokens  = tokenizer.convert_ids_to_tokens(ids)

    alive = [(i, t) for i, (t, l) in enumerate(zip(tokens, labels)) if l != -100]
    print("---------------------------------------------------------------------------")
    print(tokenizer.decode(ids))
    print("---------------------------------------------------------------------------")
    print(alive)

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    trainer.model.save_pretrained('../_final/model/phi-finetuning')

    with open(output_dir+'/train_info.json', 'w', encoding="utf-8") as make_file:
        json.dump(info_data, make_file, ensure_ascii=False, indent="\t", default=str)

    del model
    del trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    exit(main())