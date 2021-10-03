#%%
from localsearch.local_search import LocalSearchAlgorithm, NeighborhoodRelation
from typing import Callable, List

class GradientAscent(LocalSearchAlgorithm):
    def __init__(self, relation: NeighborhoodRelation, cost_fn: Callable):
        super().__init__(relation, cost_fn)

    def run(self, initialInstance, max_iter: int):
        max_sol, max_cost = initialInstance, self.cost(initialInstance, self.p1_symbol, self.p2_symbol)
        for _ in range(max_iter):
            changed = False
            neighborhood = self.relation.neighbors(max_sol)
            for solution in neighborhood:
                if (cur_cost := self.cost(solution, self.p1_symbol, self.p2_symbol)) > max_cost:
                    max_sol, max_cost = solution, cur_cost
                    changed = True
            if not changed:
                return max_sol
        return max_sol
