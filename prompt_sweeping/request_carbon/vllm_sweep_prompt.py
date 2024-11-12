import fire
import os
import random
import sys
import torch
from vllm import LLM
from vllm import LLM, SamplingParams
from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from transformers import LlamaTokenizer
import json
from carbontracker.tracker import CarbonTrackerManual
import time

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

    info = {"energy": [], "co2": [], "time": [], "num_input_tokens": [], "num_output_tokens": []}
    tracker = CarbonTrackerManual(epochs=1, monitor_epochs=1, update_interval=0.01,
    components='gpu', epochs_before_pred=1, verbose=0)
    tracker.tracker.pue_manual=1
    tracker.intensity_updater.ci_manual = 100
    time.sleep(5) # give it some cushion to initialize measurement
    
    for idx, chat in enumerate(chats):
        tracker.epoch_start()
        outputs = model.generate(prompt_token_ids=[chat], sampling_params=sampling_param)
        energy, co2, duration = tracker.epoch_end('')
        info["energy"].append(energy)
        info["co2"].append(co2)
        info["time"].append(duration)
        info["num_input_tokens"].append(len(chat))
        info["num_output_tokens"].append(len(outputs[0].outputs[0].token_ids))

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
    output_file: str="test",
    beam_search: bool=False,
    beam_size: int=1,
    prompt_limit: int=-1,
):
    # mkdir data if not exist
    from pathlib import Path
    Path("data/beam_search").mkdir(parents=True, exist_ok=True)

    model = load_model(model_name, tp_size, dtype)
    main(model, model_name=model_name, max_new_tokens=max_new_tokens, prompt_file=prompt_file, 
         top_p=top_p, temperature=temperature, seed=seed, save_path=output_file,
         beam_search=beam_search, beam_size=beam_size, prompt_limit=prompt_limit)

fire.Fire(run_script)