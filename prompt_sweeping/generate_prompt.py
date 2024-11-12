import json
import fire
import random

def main(
    prompt_file: str,
    output_name: str="prompt",
    num_prompts: int=5000,
    seed=0,
    system: bool=False,
):
    random.seed(seed)

    if "jsonl" in prompt_file:
        with open(prompt_file) as f:
            prompts = [json.loads(line) for line in f]
    else:
        with open(prompt_file) as f:
            prompts = json.load(f)
    if num_prompts > 0:
        prompts = random.sample(prompts, num_prompts)
    output = []
    for prompt in prompts:
        prompt_gen = []
        if system:
            prompt_gen.append({"role": "system", "content": "Always answer very briefly"})
        if "alpaca" in prompt_file:            
            if prompt["input"] == "":            
                prompt_gen.append({"role": "user", "content": prompt["instruction"]})
            else:
                assert prompt["instruction"] != "", "instruction cannot be empty"
                prompt_gen.append({"role": "user", "content": f"{prompt['instruction']}\n{prompt['input']}"})
        elif "grade_math" in prompt_file:
            prompt_gen.append({"role": "user", "content": prompt["question"]})
        elif "mmlu" in prompt_file:
            content = prompt["question"]
            for key, letter in zip(["answer_a", "answer_b", "answer_c", "answer_d"], ["A", "B", "C", "D"]):
                content += f"\n({letter}): {prompt[key]}"
            prompt_gen.append({"role": "user", "content": content})
        elif "helpful_base" in prompt_file:
            prompt_gen.append({"role": "user", "content": prompt["instruction"]})
        elif "triviaqa" in prompt_file or "naturalqa" in prompt_file or "scienceqa" in prompt_file:
            prompt_gen.append({"role": "user", "content": prompt["question"]})
        output.append(prompt_gen)
    addon = ""
    if system:
        addon = "_brief+"
    if num_prompts < 0:
         num_prompts = "all"
    with open(f"prompts/{output_name}_{num_prompts}{addon}.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)