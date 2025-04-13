import torch
import logging
import bisect
import math
import random
import time
from tqdm import tqdm
from typing import Callable
from src.middleware import ComponentStore
from src.assets import sgd_search_space

logger = logging.getLogger("service")

class HyperbandService:
    def __init__(self, component_store: ComponentStore = None):
        self.objective_func: Callable = None
        self.search_space: dict[str, list[float]] = None
        self._store = component_store if component_store else ComponentStore()

    def load_constrained_search_space(self, search_space: dict[str, dict[str, float]]):
        def find_nearest_index(search_list, value, lower):
            idx = bisect.bisect_left(search_list, value) if lower else bisect.bisect_right(search_list, value)
            if idx == 0:
                return 0
            if idx == len(search_list):
                return len(search_list) - 1
            return idx if abs(search_list[idx] - value) < abs(search_list[idx - 1] - value) else idx - 1

        lower = search_space["lower_bound"]
        upper = search_space["upper_bound"]

        constrained_search_space = {
            key: sgd_search_space[key][
                find_nearest_index(sgd_search_space[key], lower[key], True):
                find_nearest_index(sgd_search_space[key], upper[key], False) + 1
            ]
            for key in sgd_search_space
        }

        return constrained_search_space

    def optimise(self,
                 code_str: str,
                 search_space: dict[str, list[float]],
                 max_epochs: int = 27,
                 eta: int = 3,
                 R_key: str = "num_epochs",
                 return_score_log: bool = True) -> tuple[list[float], dict, float, list[dict]]:
        r"""
        Run Hyperband to find the best configuration.

        Params:
        - code_str: code to instantiate objective function.
        - search_space: dictionary of discrete choices.
        - max_epochs: maximum resource (e.g. num_epochs).
        - eta: downsampling rate.
        - R_key: the key in search space dict that refers to resource (e.g. 'num_epochs').
        - return_score_log: if True, return score timeline log.

        Returns:
        - List of scores,
        - Best config,
        - Best score,
        - (Optional) Score log with timestamp and config.
        """
        self.search_space = search_space
        self._store.code_string = code_str
        self._store.instantiate_code_classes()

        if not callable(self._store.objective_func):
            raise ValueError("Objective function not instantiated properly.")

        logger.info("Starting Hyperband optimisation.")
        s_max = int(math.log(max_epochs, eta))
        B = (s_max + 1) * max_epochs

        all_scores = []
        best_config = None
        best_score = float('-inf')
        score_log = []
        global_start_time = time.time()

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / max_epochs / (s + 1)) * eta ** s)
            r = max_epochs * eta ** (-s)
            configs = self._get_random_configs(n)
            logger.info(f"Bracket s={s}: {n} configs, initial resource={r}")

            for i in range(s + 1):
                n_i = n * eta ** (-i)
                r_i = int(r * eta ** i)
                logger.info(f"Round {i}: Evaluating {int(n_i)} configs with {r_i} {R_key}")

                results = []
                for config in tqdm(configs[:int(n_i)], desc=f"Hyperband round {i}"):
                    config_copy = config.copy()
                    config_copy[R_key] = r_i
                    score = self._store.objective_func(**config_copy)
                    all_scores.append(score)

                    timestamp = time.time() - global_start_time
                    score_log.append({
                        "score": float(score),
                        "config": config_copy,
                        "timestamp": timestamp
                    })

                    results.append((score, config_copy))
                    if score > best_score:
                        best_score = score
                        best_config = config_copy

                results.sort(reverse=True, key=lambda x: x[0])
                configs = [cfg for _, cfg in results[:int(n_i // eta)]]

        if return_score_log:
            return all_scores, best_config, best_score, score_log
        return all_scores, best_config, best_score

    def _get_random_configs(self, n: int) -> list[dict[str, float]]:
        keys = list(self.search_space.keys())
        space = [self.search_space[k] for k in keys]
        return [dict(zip(keys, [random.choice(dim) for dim in space])) for _ in range(n)]
