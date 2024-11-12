import json
import fire
from pathlib import Path
from glob import glob
import random

def pre_process_dataset(generation_header, data, data_name):
    if "all" not in generation_header:
        num_prompts = int(generation_header.split("_")[-1])
        data = random.sample(data, num_prompts)    
    reference = []
    if "helpful" in generation_header:
        for i in range(len(data)):
            reference.append({"instruction":data[i]["instruction"],
                            "output":data[i]["output"],
                            "generator":"gpt3",
                            "dataset":data_name})
    elif "alpaca" in generation_header:
        for i in range(len(data)):
            if data[i]["input"] == "":
                content = data[i]["instruction"]
            else:
                assert data[i]["instruction"] != "", "instruction cannot be empty"
                content = f"{data[i]['instruction']}\n{data[i]['input']}"
            reference.append({"instruction":content,
                            "output":data[i]["output"],
                            "generator":"correct_answer",
                            "dataset":data_name})
    elif "mmlu" in generation_header:
        for i in range(len(data)):
            content = data[i]["question"]
            for key, letter in zip(["answer_a", "answer_b", "answer_c", "answer_d"], ["A", "B", "C", "D"]):
                content += f"\n({letter}): {data[i][key]}"
            reference.append({"instruction":content,
                            "output":data[i]["correct_answer"],
                            "generator":"correct_answer",
                            "dataset":data_name})
    elif "naturalqa" in generation_header or "triviaqa" in generation_header:
        for i in range(len(data)):
            if len(data[i]["answer"]) == 1:
                output = data[i]["answer"][0]
            else:
                # separate answer by ";"
                output = "; ".join(data[i]["answer"])
            reference.append({"instruction":data[i]["question"],
                            "output":output,
                            "generator":"correct_answer",
                            "dataset":data_name})

    elif "math" in generation_header or "scienceqa" in generation_header:
        for i in range(len(data)):
            reference.append({"instruction":data[i]["question"],
                            "output":data[i]["answer"],
                            "generator":"correct_answer",
                            "dataset":data_name})
    return reference

def main(
    data_name: str="helpful_base", # dataset name, e.g., mmlu, helpful_base
    num_samples: int=-1,  # -1 means all
    generation_header: str="helpful_all", # generated data header, e.g., mmlu_5000, helpful_all
    seed: int=0,
    skip_beam_search: bool=True
):
    random.seed(seed)

    DATA_PATH = glob(f"../../data/{data_name}.json*")[0]
    if "jsonl" in DATA_PATH:
        with open(DATA_PATH) as f:
            data = [json.loads(line) for line in f]
    else:
        with open(DATA_PATH, 'r') as json_file:
            # Load the JSON data from the file
            data = json.load(json_file)

    # make dir data/auto_annotate if not exist
    Path(f"auto_annotate/{data_name}").mkdir(parents=True, exist_ok=True)
    if num_samples > 0:
        Path(f"auto_annotate/{data_name}/samples_{num_samples}_seed_{seed}").mkdir(parents=True, exist_ok=True)

    reference = pre_process_dataset(generation_header, data, data_name)

    if skip_beam_search:
        paths = glob(f"data/{generation_header}_*.json")
    else:
        paths = glob(f"data/beam_search/width_*_{generation_header}_*.json") + glob(f"data/{generation_header}_*.json")
    for p, path in enumerate(paths):
        with open(path, 'r') as json_file:
            # Load the JSON data from the file
            data_gen = json.load(json_file)
        if p == 0:
            if num_samples > 0:
                random_idx = random.sample(range(len(data_gen['input_text'])), num_samples)
            else:
                random_idx = range(len(data_gen['input_text']))
        generator_path = path.split("/")[-1].split("-")[0]
        output = []
        for i in range(len(data_gen['input_text'])):
            assert data_gen['input_text'][i] == reference[i]["instruction"], "Prompt mismatch"
            output.append({"instruction":data_gen['input_text'][i],
                            "output":data_gen["output_text"][i],
                            "generator":generator_path,
                            "dataset":data_name})
        if num_samples > 0:
            # select output by random_idx
            output = [output[j] for j in random_idx]
        dump_path = f"auto_annotate/{data_name}/{generator_path}.json" if num_samples == -1 \
            else f"auto_annotate/{data_name}/samples_{num_samples}_seed_{seed}/{generator_path}.json"
        with open(dump_path, "w") as f:
            json.dump(output, f, indent=4)    

    save_name = "alpaca_gpt3" if "helpful" in generation_header else "correct_answer"
    dump_path = f"auto_annotate/{data_name}/{save_name}.json" if num_samples == -1 \
        else f"auto_annotate/{data_name}/samples_{num_samples}_seed_{seed}/{save_name}.json"
    if num_samples > 0:
        # select reference by random_idx
        reference = [reference[j] for j in random_idx]
    with open(dump_path, "w") as f:
        json.dump(reference[:len(output)], f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)