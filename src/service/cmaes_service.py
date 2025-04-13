import torch
import logging
import bisect
import numpy as np
from typing import Callable
from src.middleware import ComponentStore
from src.assets import sgd_search_space
import cma

logger = logging.getLogger("service")

class CMAESService:
    def __init__(self, component_store: ComponentStore = None):
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
                 sigma: float = 0.5,
                 max_iter: int = 100,
                 population_size: int = 10) -> tuple[list[float], dict, float]:
        r"""
        Run CMA-ES to find the best configuration.

        Params:
        - code_str: code to instantiate objective function.
        - search_space: dictionary of discrete values per parameter.
        - sigma: standard deviation (controls initial exploration).
        - max_iter: maximum number of CMA-ES iterations.
        - population_size: number of samples per generation.

        Returns:
        - List of scores,
        - Best config,
        - Best score.
        """
        self.search_space = search_space
        self._store.code_string = code_str
        self._store.instantiate_code_classes()

        if not callable(self._store.objective_func):
            raise ValueError("Objective function not instantiated properly.")

        logger.info("Starting CMA-ES optimisation.")
        all_scores = []

        keys = list(search_space.keys())
        index_space = [list(range(len(search_space[k]))) for k in keys]
        bounds = [[0] * len(keys), [len(dim) - 1 for dim in index_space]]

        x0 = [len(dim) // 2 for dim in index_space]  # Start from mid-point
        es = cma.CMAEvolutionStrategy(x0=x0, sigma=sigma, inopts={
            'bounds': bounds,
            'popsize': population_size,
            'maxiter': max_iter,
            'verb_disp': 0,
        })

        best_score = float('-inf')
        best_config = None

        while not es.stop():
            solutions = es.ask()
            scores = []
            for solution in solutions:
                indices = np.round(np.clip(solution, 0, np.array(bounds[1]))).astype(int)
                params = {k: self.search_space[k][idx] for k, idx in zip(keys, indices)}
                try:
                    score = self._store.objective_func(**params)
                except Exception as e:
                    logger.warning(f"Objective function failed: {e}")
                    score = -1e6  # Penalise invalid configs

                scores.append(-score)  # CMA-ES minimises; we maximise
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_config = params

            es.tell(solutions, scores)

        return all_scores, best_config, best_score
