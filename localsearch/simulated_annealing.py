#%%
from localsearch.local_search import LocalSearchAlgorithm, NeighborhoodRelation
from typing import Callable
import networkx as nx
import random
import math

class SimulatedAnnealing(LocalSearchAlgorithm):
    def __init__(self, relation: NeighborhoodRelation, cost_fn: Callable):
        super().__init__(relation, cost_fn)
    
    def run(self,
            initialInstance: nx.Graph,
            max_iter: int, 
            cooling_schedule: Callable,
            constant: int):
        max_sol, max_cost = initialInstance, self.cost(initialInstance, self.p1_symbol, self.p2_symbol)
        cur_sol, cur_cost = max_sol, max_cost
        for i in range(max_iter):
            neighborhood = self.relation.neighbors(cur_sol)
            for solution in neighborhood:
                if (c := self.cost(solution, self.p1_symbol, self.p2_symbol)) >= cur_cost:
                    cur_sol, cur_cost = solution, c
                    if cur_cost > max_cost:
                        max_sol, max_cost = cur_sol, cur_cost
                else:
                    delta_cost = abs(c - cur_cost)
                    if random.random() < math.exp(-1 * (delta_cost / (cooling_schedule(i+1) * constant))):
                        cur_sol, cur_cost = solution, c
                        break
        return max_sol
