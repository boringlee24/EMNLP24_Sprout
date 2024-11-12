import json
import numpy as np
from pathlib import Path
import pandas as pd
from glob import glob
import random
import math

def load_carbon_intensity(
    region: str="US-CAL",
    start_hour: int=800,
    interval: int=30*24,
):
    region_data = glob(f"{Path(__file__).parents[1]}/carbon_intensity/{region}*_2023_hourly.csv")[0]
    df = pd.read_csv(region_data)
    data = df["Carbon Intensity gCO₂eq/kWh (direct)"].values[start_hour:start_hour+interval]
    ci_max = df["Carbon Intensity gCO₂eq/kWh (direct)"].max()
    ci_min = df["Carbon Intensity gCO₂eq/kWh (direct)"].min()
    return data.tolist(), ci_max, ci_min

class OfflineAssessor:
    def __init__(self,
                 ci: list,
                 decay: float=0.029, # half confidence after 24 hrs
                 grace_period: int=6,
                 max_idle_time: int=24*3
                 ) -> None:
        self.ci = ci
        self.history_avg = np.mean(ci)
        self.decay = decay
        self.last_update_idx = 0
        self.gradients = [0, 0]
        self.prev_time = 0
        self.grace_period = grace_period
        self.selected_time = []
        self.max_idle_time = max_idle_time

    def perform_offline_assessment(
        self,
        time_index: int,
    ):       
        if time_index == 0:
            return False
        
        assert time_index == self.prev_time + 1, "Time index is not continuous."
        self.prev_time = time_index

        self.gradients[0] = self.gradients[1]
        recorded_value = []
        for index in [time_index-1, time_index]:
            multipler = math.exp(-self.decay * (index-self.last_update_idx))
            recorded_value.append(self.ci[index] / multipler)
        gradient = recorded_value[1] - recorded_value[0]
        self.gradients[1] = gradient
        if time_index - self.last_update_idx >= self.max_idle_time:
            self.last_update_idx = time_index
            self.selected_time.append(time_index)
            return True
        elif time_index - self.last_update_idx <= self.grace_period:
            return False
        elif self.ci[time_index] > self.history_avg:
            return False
        elif self.gradients[0] <= 0 and self.gradients[1] >= 0:
            self.last_update_idx = time_index
            self.selected_time.append(time_index)
            return True
        else:
            return False

def get_quality(
    sample_dict: dict, # {dataset_name: [prompt_1, prompt_2, ...]}
    reference_path: str,
    num_lvls: int=3,
):
    # init each lvl pref rate with 0, store as dict
    pref_count = {i: 0 for i in range(num_lvls)}
    # from copy import deepcopy #TODO
    # pref_count_dataset = {k: deepcopy(pref_count) for k in sample_dict} # this is just for test purpose

    preference = {}
    with open(reference_path) as f:
        reference = json.load(f)
        for k, v in reference.items():
            preference[k] = {i["instruction"]: i["assistance_lvl"] for i in v}
    for dataset in sample_dict:
        for prompt in sample_dict[dataset]:
            pref_count[preference[dataset][prompt]] += 1
            # pref_count_dataset[dataset][preference[dataset][prompt]] += 1

    score = {}
    for k, v in pref_count.items():
        score[k] = v / sum(pref_count.values())
    return score

def get_per_prompt_data_for_oracle(
    sample_dict: dict, # {dataset_name: [prompt_1, prompt_2, ...]}
    reference_path: str,
    num_lvls: int=3,        
):
    output = []
    preference = {}

    with open(reference_path) as f:
        reference = json.load(f)
        for k, v in reference.items():
            preference[k] = {i["instruction"]: i["assistance_lvl"] for i in v}
    with open(f"{Path(__file__).parents[2]}/prompt_sweeping/online_generation/data/summary.json") as f:
        read = json.load(f)

    for dataset in sample_dict:
        for prompt in sample_dict[dataset]:
            # quality = [0] * num_lvls
            # quality[preference[dataset][prompt]] = 1
            output.append({
                "quality": preference[dataset][prompt],
                "energy": read[dataset][prompt]["energy"],
                "time": read[dataset][prompt]["time"]
            })

    return output

def get_energy_and_time_for_oracle(
    sample_dict: dict, # {dataset_name: [prompt_1, prompt_2, ...]}
    lvls: dict={0: "13b", 1: "brief_13b", 2: "brief+_13b"},
):
    energy, time = {}, {}
    with open(f"{Path(__file__).parents[2]}/prompt_sweeping/online_generation/data/summary.json") as f:
        read = json.load(f)

    for lvl in lvls:
        energy[lvl] = []
        time[lvl] = []
    for dataset in sample_dict:
        for prompt in sample_dict[dataset]:
            for lvl in lvls:
                energy[lvl].append(read[dataset][prompt]["energy"][lvl])
                time[lvl].append(read[dataset][prompt]["time"][lvl])
    for lvl in lvls:
        energy[lvl] = np.mean(energy[lvl])
        time[lvl] = np.mean(time[lvl])
    return energy, time

def get_energy_and_time(
    datasets: list=["alpaca", "math", "mmlu", "naturalqa", "triviaqa"],
    composition: list=[0.2, 0.2, 0.2, 0.2, 0.2],
    num_prompts: int=10000,
    lvls: dict={0: "13b", 1: "brief_13b", 2: "brief+_13b"},
):
    """
    Under current probability configuration of prompt assistance, 
    take 10000 samples, for each sample, assign its assist level get its carbon emission.
    Finally, return the average carbon emission of all prompt levels 
    (embodied + operational) on this current dataset composition.
    returns:
        {0: avg_carbon_0, 1: avg_carbon_1, 2: avg_carbon_2}
    """
    dataset_choice = np.random.choice(datasets, p=composition, size=num_prompts)
    energy, time = {}, {}
    for lvl in lvls:
        energy[lvl] = []
        time[lvl] = []
    for dataset in datasets:
        count = np.sum(dataset_choice == dataset)         
        name = f"{dataset}_5000" if dataset != "math" else "math_all"
        # randomly choose count number of index from 0 to 5000, repetition allowed
        sample_idx = random.choices(range(5000), k=count)        
        for lvl, config in lvls.items():
            with open(f"{Path(__file__).parents[2]}/prompt_sweeping/online_generation/data/{name}_{config}-chat.json") as f:
                read = json.load(f)
            # select based on sample_idx
            energy[lvl] += [read["energy"][i] for i in sample_idx]
            time[lvl] += [read["time"][i] for i in sample_idx]
    assert len(energy[0]) == num_prompts and len(time[0]) == num_prompts, "Energy and time length mismatch."
    for lvl in lvls:
        energy[lvl] = np.mean(energy[lvl])
        time[lvl] = np.mean(time[lvl])
    return energy, time

class CarbonEvaluator:
    def __init__(self,
                embodied_carbon_per_sec: float,
                datasets: list=["alpaca", "math", "mmlu", "naturalqa", "triviaqa"],
                lvls: dict={0: "13b", 1: "brief_13b", 2: "brief+_13b"},
                ) -> None:
        self.embodied_carbon_per_sec = embodied_carbon_per_sec
        self.lookup = {}
        self.lvls = list(lvls.keys())
        self.datasets = datasets
        for dataset in datasets:
            self.lookup[dataset] = {}
            name = f"{dataset}_5000" if dataset != "math" else "math_all"
            for lvl, config in lvls.items():
                with open(f"{Path(__file__).parents[2]}/prompt_sweeping/online_generation/data/{name}_{config}-chat.json") as f:
                    read = json.load(f)
                self.lookup[dataset][lvl] = read

    def eval_carbon(self, x: list, ci: float, num_prompts: int=10000, composition: list=[0.2, 0.2, 0.2, 0.2, 0.2]):
        dataset_choice = np.random.choice(self.datasets, p=composition, size=num_prompts)
        num_data = len(self.lookup[self.datasets[0]][0]["energy"])
        total_carbon = 0
        for data_choice in dataset_choice:
            # sample the lvl based on probability from x
            lvl = np.random.choice(self.lvls, p=x)
            # sample the prompt
            prompt_idx = random.choice(range(num_data))
            energy = self.lookup[data_choice][lvl]["energy"][prompt_idx]
            time_spent = self.lookup[data_choice][lvl]["time"][prompt_idx]
            total_carbon += energy * ci + self.embodied_carbon_per_sec * time_spent
        return total_carbon / num_prompts

"""
generate combined prompts for all datasets
"""
class PromptGenerator:
    def __init__(self, 
                 seed: int=0,
                 datasets: list=["alpaca", "math", "mmlu", "naturalqa", "triviaqa"],
                 interval: int=30*24, # number of hours
                 update_freq: int=24, # number of hours per update
                 ):
        random.seed(seed)     
        np.random.seed(seed)   
        self.datasets = datasets
        self.update_freq = update_freq
        self.update_index_random = 0
        self.composition = [[]]*interval
        self.generation_index = 0

    def generate_random(self):
        """
        Generate a prompt configuration for each time stamp
        """
        if self.generation_index == self.update_index_random:
            random_numbers = np.random.randint(0, 101, len(self.datasets))
            normalized_numbers = random_numbers / random_numbers.sum()
            # # Adjust the last element
            # normalized_numbers[-1] = 1 - normalized_numbers[:-1].sum()
            self.composition[self.generation_index] = normalized_numbers.tolist()
            self.update_index_random += int(round(np.random.normal(self.update_freq, 0.2*self.update_freq)))
            assert np.isclose(sum(self.composition[self.generation_index]), 1, atol=1e-8), "Prompt composition does not sum up to 1."
        else:
            self.composition[self.generation_index] = self.composition[self.generation_index-1]
        self.generation_index += 1

    def generate_all(self):
        """
        Generate all prompt configurations
        """
        for i in range(len(self.composition)):
            self.generate_random()

    def get_prompts(self, time_index, num_prompts: int=1000, testcase: str="samples_2000", repetition=False):
        """
        Get the prompts for each dataset.
        Performs prompt lookup in the dataset/samples_2000_seed_0 dir
        """
        prompts = {}
        # sample the dataset based on probability
        dataset_choice = np.random.choice(self.datasets, p=self.composition[time_index], size=num_prompts)
        # count the number of prompts for each dataset
        for dataset in self.datasets:
            count = np.sum(dataset_choice == dataset)
            with open(f"{Path(__file__).parents[2]}/prompt_sweeping/batched_inference/auto_annotate/{dataset}/{testcase}_seed_0/correct_answer.json") as f:
                used_prompts = json.load(f)
            if repetition:
                sampled_prompts = random.choices(used_prompts, k=count)
            else:
                sampled_prompts = random.sample(used_prompts, count)
            prompts[dataset] = list(map(lambda x: x["instruction"], sampled_prompts))
        return prompts

    def print_stats(self):
        """
        Print the statistics of the prompt composition
        """
        average = [sum(x) / len(x) for x in zip(*self.composition)]
        for i, dataset in enumerate(self.datasets):
            print(f"{dataset}: {average[i]}")
        self.overall_composition = average

class PromptGeneratorPAI(PromptGenerator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pai_df = pd.read_csv(f"{Path(__file__).parents[0]}/pai.csv").job_name
        self.trace_starting_index = {}
        for dataset in self.datasets:
            starting_index = np.random.randint(0, len(self.pai_df)-len(self.composition)/self.update_freq)
            self.trace_starting_index[dataset] = starting_index
            # print(f"{dataset}: {np.mean(self.pai_df[starting_index:starting_index+self.update_freq])}")

    def generate_random(self):
        """
        Generate a prompt configuration for each time stamp
        """

        if self.generation_index == self.update_index_random:
            selected_numbers = []
            for dataset in self.datasets:
                starting_index = self.trace_starting_index[dataset]
                selected_numbers.append(self.pai_df[starting_index])
                self.trace_starting_index[dataset] += 1
            normalized_numbers = selected_numbers / sum(selected_numbers)
            self.composition[self.generation_index] = normalized_numbers.tolist()
            self.update_index_random += int(round(np.random.normal(self.update_freq, 0.2*self.update_freq)))
            assert np.isclose(sum(self.composition[self.generation_index]), 1, atol=1e-8), "Prompt composition does not sum up to 1."
        else:
            self.composition[self.generation_index] = self.composition[self.generation_index-1]
        self.generation_index += 1

if __name__ == "__main__":
    gen = PromptGeneratorPAI()
    gen.generate_all()
    gen.print_stats()
    prompts = gen.get_prompts(0, num_prompts=5000)
    aa = get_quality(prompts, f"{Path(__file__).parents[2]}/prompt_sweeping/batched_inference/auto_annotate/quality_accessor/assessment_results/summary_2000.json")#f"{Path(__file__).parent}/summary_2000.json")
    print(aa)
