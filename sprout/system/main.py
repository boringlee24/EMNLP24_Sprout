import fire
import utils
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from optimizer import BaseOptimizer, LPOptimizer, CloverOptimizer, CO2Optimizer, StaticOptimizer, Oracle, NoEvalOptimizer

def main(
    region: str="US-CAL",
    start_hour: int=800,
    interval: int=30*24,
    datasets: list=["alpaca", "math", "mmlu", "naturalqa", "triviaqa", "scienceqa"],
    generator: str="pai", #"random",
    optimizer_name: str="lp",
    pref_torelance: float=0.1,
    seed: int=0,
    decay: float=0.029,
    num_sample_prompts: int=1000,
    embodied_carbon: float=26356.568,
    device_life_years: int=5,
    grace_period: int=6,
    update_freq: int=24,
):
    # load carbon intensity data
    carbon_intensity, ci_max, ci_min = utils.load_carbon_intensity(region, start_hour, interval)
    embodied_carbon_per_sec=embodied_carbon / device_life_years / 365 / 24 / 3600
    offline_assessor = utils.OfflineAssessor(
        ci = carbon_intensity,
        decay = decay,
        grace_period = grace_period
    )

    if generator == "random":
        gen = utils.PromptGenerator(seed=seed,
                                    datasets=datasets,
                                    interval=interval,
                                    update_freq=update_freq)
    elif generator == "pai":
        gen = utils.PromptGeneratorPAI(seed=seed,
                                    datasets=datasets,
                                    interval=interval,
                                    update_freq=update_freq)
    else:
        raise ValueError("Invalid generator")
    gen.generate_all()

    if optimizer_name == "baseline":
        optimizer_obj = BaseOptimizer(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
        )
    elif optimizer_name == "lp":
        optimizer_obj = LPOptimizer(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
        )
    elif optimizer_name == "no_eval":
        optimizer_obj = NoEvalOptimizer(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
        )        
    elif optimizer_name == "clover":
        optimizer_obj = CloverOptimizer(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
        )
    elif optimizer_name == "static":
        ci_past_month,_,_ = utils.load_carbon_intensity(region, 0, 8759) #start_hour-interval, interval)
        optimizer_obj = StaticOptimizer(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
            interval=interval,
            carbon_intensity=ci_past_month,
        )
    elif optimizer_name == "co2opt":
        optimizer_obj = CO2Optimizer(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
        )
    elif optimizer_name == "oracle":
        optimizer_obj = Oracle(
            datasets=datasets,
            optimizer=optimizer_name,
            prompt_gen=gen,
            embodied_carbon_per_sec=embodied_carbon_per_sec,
            ci_min=ci_min,
            ci_max=ci_max,
            pref_torelance=pref_torelance,
            offline_assessor=offline_assessor,
        )
    else:
        raise ValueError("Invalid optimizer")
    for hour, ci in tqdm(enumerate(carbon_intensity), total=len(carbon_intensity)):
        optimizer_obj.optimize_setup(
            hour=hour,
            num_sample_prompts=num_sample_prompts
        )
        optimizer_obj.optimize(ci=ci, hour=hour)

    Path(f"{Path(__file__).resolve().parent}/results/{region}/season_{start_hour}/torelance_{pref_torelance}").mkdir(parents=True, exist_ok=True)
    with open(f"{Path(__file__).resolve().parent}/results/{region}/season_{start_hour}/torelance_{pref_torelance}/{optimizer_name}_opt_{generator}_gen.json", "w") as f:
        json.dump(optimizer_obj.result, f, indent=4)
    if optimizer_name == "lp":
        with open(f"{Path(__file__).resolve().parent}/results/{region}/season_{start_hour}/offline_assessment.json", "w") as f:
            json.dump(offline_assessor.selected_time, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)

