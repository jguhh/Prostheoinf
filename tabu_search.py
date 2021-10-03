# %%
from localsearch.local_search import LocalSearchAlgorithm, NeighborhoodRelation
from typing import Callable

class TabuSearch(LocalSearchAlgorithm):
    def __init__(self, relation: NeighborhoodRelation, cost_fn: Callable):
        super().__init__(relation, cost_fn)
        self.tabu_list = []

    def run(self, initialInstance, max_iter: int):
        max_sol, max_cost = initialInstance, self.cost(initialInstance, self.p1_symbol, self.p2_symbol)
        cur_sol, cur_cost = max_sol, max_cost
        self.tabu_list.append(cur_sol)
        
        for _ in range(max_iter):
            changed = False
            neighborhood = self.relation.neighbors(cur_sol)

            neighbor_sol, neighbor_cost = None, -1
            for solution in neighborhood:
                if (c := self.cost(solution, self.p1_symbol, self.p2_symbol)) > neighbor_cost\
                        and solution not in self.tabu_list:
                    neighbor_sol, neighbor_cost = solution, c
                    changed = True
                    self.tabu_list.append(solution)
            cur_sol, cur_cost = neighbor_sol, neighbor_cost
            if neighbor_cost > max_cost:
                max_sol, max_cost = cur_sol, cur_cost
            
            if not changed:
                return max_sol
        return max_sol
