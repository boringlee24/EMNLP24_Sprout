from typing import Optional
import fire
import os
import random
import time
import sys
import torch
from vllm import LLM
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from transformers import LlamaTokenizer
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Optional, Union
from pathlib import Path

def load_model(model_name, tp_size=1, dtype="auto"):

    llm = LLM(model_name, 
                    tensor_parallel_size=tp_size, 
                    dtype=dtype,
                    max_num_batched_tokens=4096, 
                    max_model_len=4096,
                    swap_space=16,
                    gpu_memory_utilization=0.95,
                    )
    return llm

def main(
    model,
    model_name,
    max_new_tokens=2048,
    prompt_file: str=None,
    top_p=1.0,
    temperature=1.0,
    seed=0,
    top_k = 50,
    save_path: str="test",
    beam_search: bool=False,
    beam_size: int=1,
    beam_batch_size: int=8,
    prompt_limit: int=-1,
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)
        if prompt_limit > 0:
            dialogs = dialogs[:prompt_limit]
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.set_tokenizer(tokenizer)
    chats = format_tokens(dialogs, tokenizer)
    best_of = beam_size if beam_search else 1
    temperature = 0 if beam_search else temperature
    top_k = -1 if beam_search else top_k
    sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens, top_k=top_k, use_beam_search=beam_search, best_of=best_of)
    
    info = {}
    if not beam_search:
        outputs = model.generate(prompt_token_ids=chats, sampling_params=sampling_param)    
    else:
        batch_size = beam_batch_size
        outputs = []
        for i in tqdm(range(0, len(chats), batch_size)):
            batch = chats[i:i+batch_size]
            outputs.extend(model.generate(prompt_token_ids=batch, sampling_params=sampling_param))
    info["input_text"] = [dialog[-1]["content"] for dialog in dialogs]
    info["output_text"] = [output.outputs[0].text for output in outputs]
    info["num_output_tokens"] = [len(output.outputs[0].token_ids) for output in outputs]

    if beam_search:
        save_path = f"beam_search/width_{beam_size}_{save_path}"
    with open(f"data/{save_path}.json", "w") as f:
        json.dump(info, f, indent=4)

def run_script(
    model_name: str,
    tp_size=1,
    max_new_tokens=2048,
    prompt_file: str=None,
    top_p=1.0,
    temperature=1.0,
    seed=0,
    dtype: str="auto",
    save_path: str="test",
    beam_search: bool=False,
    beam_size: int=1,
    beam_batch_size: int=8,
    prompt_limit: int=-1,
):
    # mkdir data if not exist
    Path("data/beam_search").mkdir(parents=True, exist_ok=True)

    model = load_model(model_name, tp_size, dtype)
    main(model, model_name=model_name, max_new_tokens=max_new_tokens, prompt_file=prompt_file, 
         top_p=top_p, temperature=temperature, seed=seed, save_path=save_path,
         beam_search=beam_search, beam_size=beam_size, beam_batch_size=beam_batch_size,
         prompt_limit=prompt_limit)

fire.Fire(run_script)