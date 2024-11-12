import json
import numpy as np
from pathlib import Path
import pandas as pd
from glob import glob
import random
from utils import PromptGenerator
import utils
from scipy.optimize import linprog

class BaseOptimizer:
    def __init__(self,                  
                 datasets: list,
                 optimizer: str,
                 prompt_gen: PromptGenerator,
                 embodied_carbon_per_sec: float,
                 ci_min: float,
                 ci_max: float,
                 pref_torelance: float,
                 offline_assessor: utils.OfflineAssessor,
                 ) -> None:
        if optimizer == "clover":
            self.reference_path = f"{Path(__file__).resolve().parents[2]}/prompt_sweeping/batched_inference/auto_annotate/quality_accessor/assessment_results/vs_7b/summary_2000.json"
            self.num_lvls = 2
            self.lvls = {0: "13b", 1: "7b"}
        else:
            self.reference_path = f"{Path(__file__).resolve().parents[2]}/prompt_sweeping/batched_inference/auto_annotate/quality_accessor/assessment_results/summary_2000.json"
            self.num_lvls = 3
            self.lvls = {0: "13b", 1: "brief_13b", 2: "brief+_13b"}
        self.optimizer = optimizer
        self.datasets = datasets
        self.prompt_gen = prompt_gen
        self.result = {}
        for key in ["ci", "x", "carbon_per_req", "quality_preference", "pref_max"]:
            self.result[key] = []
        self.embodied_carbon_per_sec = embodied_carbon_per_sec
        self.ci_min, self.ci_max = ci_min, ci_max
        self.pref_torelance = pref_torelance
        self.evaluator = utils.CarbonEvaluator(
            embodied_carbon_per_sec=self.embodied_carbon_per_sec,
            datasets=self.datasets,
            lvls=self.lvls,
        )
        self.offline_assessor = offline_assessor

    def update_quality(self, hour: int):
        self.quality = self.true_quality

    def optimize_setup(self,
                 hour: int,
                 num_sample_prompts: int=1000,
                 ):
        self.sampled_prompts = self.prompt_gen.get_prompts(time_index=hour, num_prompts=num_sample_prompts)
        self.true_quality = utils.get_quality(
            sample_dict=self.sampled_prompts,
            reference_path=self.reference_path,
            num_lvls=self.num_lvls
        )
        self.update_quality(hour=hour)

        self.energy_per_lvl, self.time_per_lvl = utils.get_energy_and_time(
            datasets=self.datasets,
            composition=self.prompt_gen.composition[hour],
            num_prompts=10000,
            lvls=self.lvls
        )

    def optimize(self,
                 hour: int,
                 ci: float,):
        self.result["ci"].append(ci)
        x = [1] + [0]*(self.num_lvls-1)
        self.result["x"].append(x)            
        # self.result["carbon_per_req"].append(carbon_per_req)
        self.result["carbon_per_req"].append(
            self.evaluator.eval_carbon(
                x=self.result["x"][-1],
                ci=ci,
                num_prompts=10000,
                composition=self.prompt_gen.composition[hour]
        ))
        self.result["quality_preference"].append(self.true_quality[0])
        self.result["pref_max"].append(self.true_quality[0])
        
class LPOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def update_quality(self, hour: int):
        if self.offline_assessor.perform_offline_assessment(time_index=hour) or hour == 0:
            self.quality = self.true_quality

    def optimize(self,
                 hour: int,
                 ci: float,):
        c_vector = []
        for i in range(self.num_lvls):
            c_vector.append(self.energy_per_lvl[i] * ci + self.embodied_carbon_per_sec * self.time_per_lvl[i])
        A_ub = [[-k for k in self.quality.values()]]
        b_ub_scale = 1 - (ci-self.ci_min) / (self.ci_max-self.ci_min) * self.pref_torelance
        b_ub = [-b_ub_scale*self.quality[0]]
        A_eq = [[1]*self.num_lvls]
        b_eq = 1
        bounds = [(0,1)]*self.num_lvls
        res = linprog(c_vector, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        self.result["ci"].append(ci)
        self.result["x"].append(res.x.tolist())
        self.result["carbon_per_req"].append(
            self.evaluator.eval_carbon(
                x=self.result["x"][-1],
                ci=ci,
                num_prompts=10000,
                composition=self.prompt_gen.composition[hour]
        ))
        self.result["quality_preference"].append(np.dot(list(self.true_quality.values()), res.x))
        self.result["pref_max"].append(self.true_quality[0])

class NoEvalOptimizer(LPOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def update_quality(self, hour: int):
        if hour == 0:
            self.quality = self.true_quality
        # self.quality = {0: 0.7, 1: 0.2, 2: 0.1}

class CloverOptimizer(LPOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self,
                 hour: int,
                 ci: float,):
        # since it's configuring model variant, it cannot have infinity granularity (x is not a probability any more)
        c_vector = []
        for i in range(self.num_lvls):
            c_vector.append(self.energy_per_lvl[i] * ci + self.embodied_carbon_per_sec * self.time_per_lvl[i])
        A_ub = [[-k for k in self.quality.values()]]
        b_ub_scale = 1 - (ci-self.ci_min) / (self.ci_max-self.ci_min) * self.pref_torelance
        b_ub = [-b_ub_scale*self.quality[0]]
        A_eq = [[1]*self.num_lvls]
        b_eq = 1
        bounds = [(0,1)]*self.num_lvls
        res = linprog(c_vector, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        x = [round(res.x[0], 1), 1-round(res.x[0], 1)]
        self.result["ci"].append(ci)
        self.result["x"].append(x)
        # self.result["carbon_per_req"].append(np.dot(c_vector, x))
        self.result["carbon_per_req"].append(
            self.evaluator.eval_carbon(
                x=self.result["x"][-1],
                ci=ci,
                num_prompts=10000,
                composition=self.prompt_gen.composition[hour]
        ))
        self.result["quality_preference"].append(np.dot(list(self.true_quality.values()), x))
        self.result["pref_max"].append(self.true_quality[0])

class CO2Optimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self,
                 hour: int,
                 ci: float,):
        # find the lest energy consuming level
        co2opt_lvl = min(self.energy_per_lvl, key=lambda k: np.mean(self.energy_per_lvl[k]) * ci + self.embodied_carbon_per_sec * self.time_per_lvl[k])
        # carbon_per_req = self.energy_per_lvl[co2opt_lvl] * ci + self.embodied_carbon_per_sec * self.time_per_lvl[co2opt_lvl]
        self.result["ci"].append(ci)
        x = []
        for i in range(self.num_lvls):
            if i == co2opt_lvl:
                x.append(1)
            else:
                x.append(0)
        self.result["x"].append(x)            
        self.result["carbon_per_req"].append(
            self.evaluator.eval_carbon(
                x=self.result["x"][-1],
                ci=ci,
                num_prompts=10000,
                composition=self.prompt_gen.composition[hour]
        ))
        self.result["quality_preference"].append(self.true_quality[co2opt_lvl])
        self.result["pref_max"].append(self.true_quality[0])

class StaticOptimizer(BaseOptimizer):
    def __init__(self, interval: int, carbon_intensity: list, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # calculate optimal static x
        sampled_prompts_list = []
        for time_index in range(interval):
            sampled_prompts_list.append(self.prompt_gen.get_prompts(time_index=time_index, num_prompts=1000))
        sampled_prompts = {key: [] for key in sampled_prompts_list[0]}
        for d in sampled_prompts_list:
            for key, value in d.items():
                sampled_prompts[key].extend(value)
        quality = utils.get_quality(
            sample_dict=sampled_prompts,
            reference_path=self.reference_path,
            num_lvls=self.num_lvls
        )

        energy_lvl_list, time_lvl_list = [], []
        for time_index in range(interval):
            energy_per_lvl, time_per_lvl = utils.get_energy_and_time(
                datasets=self.datasets,
                composition=self.prompt_gen.composition[time_index],
                num_prompts=10000,
                lvls=self.lvls
            )        
            energy_lvl_list.append(energy_per_lvl)
            time_lvl_list.append(time_per_lvl)
        energy_per_lvl = {key: [] for key in energy_lvl_list[0]}
        time_per_lvl = {key: [] for key in time_lvl_list[0]}
        for d1, d2 in zip(energy_lvl_list, time_lvl_list):
            for i in range(self.num_lvls):
                energy_per_lvl[i].append(d1[i])
                time_per_lvl[i].append(d2[i])
        for i in range(self.num_lvls):
            energy_per_lvl[i] = np.mean(energy_per_lvl[i])
            time_per_lvl[i] = np.mean(time_per_lvl[i])

        #  now optimize it
        c_vector = []
        for i in range(self.num_lvls):
            c_vector.append(energy_per_lvl[i] * np.mean(carbon_intensity) + self.embodied_carbon_per_sec * time_per_lvl[i])
        A_ub = [[-k for k in quality.values()]]
        b_ub_scale = 1 - (np.mean(carbon_intensity)-self.ci_min) / (self.ci_max-self.ci_min) * self.pref_torelance
        b_ub = [-b_ub_scale*quality[0]]
        A_eq = [[1]*self.num_lvls]
        b_eq = 1
        bounds = [(0,1)]*self.num_lvls
        res = linprog(c_vector, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        self.static_x = res.x.tolist()

    def optimize(self, 
                 hour: int,
                 ci: float,):
        c_vector = []
        for i in range(self.num_lvls):
            c_vector.append(self.energy_per_lvl[i] * ci + self.embodied_carbon_per_sec * self.time_per_lvl[i])

        self.result["ci"].append(ci)
        self.result["x"].append(self.static_x)
        # self.result["carbon_per_req"].append(np.dot(c_vector, self.static_x))
        self.result["carbon_per_req"].append(
            self.evaluator.eval_carbon(
                x=self.result["x"][-1],
                ci=ci,
                num_prompts=10000,
                composition=self.prompt_gen.composition[hour]
        ))
        self.result["quality_preference"].append(np.dot(list(self.true_quality.values()), self.static_x))
        self.result["pref_max"].append(self.true_quality[0])

class Oracle(BaseOptimizer):
    """
    difference to the LP optimizer
    1. it optimizes x for every request, using the request's quality/carbon/time.
    2. after optimizing x, LP optimizer evaluates the object function value using new samples, oracle does not do this.
    3. always use the true quality
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.result["true_quality"] = []
    
    def optimize_setup(self,
                 hour: int,
                 num_sample_prompts: int,
                 num_all_prompts: int=10000,
                 ):
        """
        collect 10000 prompts for the hour
        for each prompt, collect its true quality, energy and time per level
        """
        self.sampled_prompts = self.prompt_gen.get_prompts(time_index=hour, num_prompts=num_all_prompts, repetition=True)

        self.true_quality = utils.get_quality(
            sample_dict=self.sampled_prompts,
            reference_path=self.reference_path,
            num_lvls=self.num_lvls
        )
        self.quality = self.true_quality
        self.energy_per_lvl, self.time_per_lvl = utils.get_energy_and_time_for_oracle(
            sample_dict=self.sampled_prompts,
            lvls=self.lvls
        )

        self.per_prompt_data = utils.get_per_prompt_data_for_oracle(
            sample_dict=self.sampled_prompts,
            reference_path=self.reference_path,
            num_lvls=self.num_lvls
        )
    def optimize(self,
                 hour: int,
                 ci: float,):
        """
        optimize x using LP
        then based on probability, sort request by their carbon emission
        say x = [0.8, 0, 0.2]
        the requests with highest 20% of lvl2 to lvl0 carbon savings will be assigned to level 2 
        the rest assigned to lvl0
        """
        c_vector = []
        for i in range(self.num_lvls):
            c_vector.append(self.energy_per_lvl[i] * ci + self.embodied_carbon_per_sec * self.time_per_lvl[i])
        A_ub = [[-k for k in self.quality.values()]]
        b_ub_scale = 1 - (ci-self.ci_min) / (self.ci_max-self.ci_min) * self.pref_torelance
        b_ub = [-b_ub_scale*self.quality[0]]
        A_eq = [[1]*self.num_lvls]
        b_eq = 1
        bounds = [(0,1)]*self.num_lvls
        res = linprog(c_vector, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

        # find the two carbon saving lvls that are active from res.x
        active_lvls = []
        for i in range(len(res.x)):
            if res.x[i] > 0:
                active_lvls.append(i)
        assert len(active_lvls) == 2, "only two lvls can be active"

        # create carbon emission list
        carbon_emission = []
        for prompt in self.per_prompt_data:
            record = {}
            for lvl in active_lvls:
                record[lvl] = prompt["energy"][lvl] * ci + self.embodied_carbon_per_sec * prompt["time"][lvl]
            record["saving"] = record[active_lvls[0]] - record[active_lvls[1]]
            record["quality"] = prompt["quality"]
            carbon_emission.append(record)
        carbon_per_request_sorted = sorted(carbon_emission, key=lambda k: k["saving"], reverse=True)
        total_carbon = 0
        quality_preference = 0
        for i in range(len(carbon_per_request_sorted)):
            if i < res.x[active_lvls[-1]] * len(self.per_prompt_data):
                total_carbon += carbon_per_request_sorted[i][active_lvls[1]]
                if carbon_per_request_sorted[i]["quality"] == active_lvls[-1]:
                    quality_preference += 1
            else:  
                total_carbon += carbon_per_request_sorted[i][active_lvls[0]]
                if carbon_per_request_sorted[i]["quality"] == active_lvls[0]:
                    quality_preference += 1
        self.result["ci"].append(ci)
        self.result["x"].append(res.x.tolist())
        self.result["carbon_per_req"].append(total_carbon/len(self.per_prompt_data))
        self.result["quality_preference"].append(np.dot(list(self.true_quality.values()), res.x))
        self.result["pref_max"].append(self.true_quality[0])
        self.result["true_quality"].append(quality_preference/len(self.per_prompt_data))
