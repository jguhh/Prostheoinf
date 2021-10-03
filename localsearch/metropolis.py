#%%
from localsearch.local_search import NeighborhoodRelation
from localsearch.simulated_annealing import SimulatedAnnealing
from typing import Callable
import networkx as nx

class Metropolis(SimulatedAnnealing):
    def __init__(self, relation: NeighborhoodRelation, cost_fn: Callable):
        super().__init__(relation, cost_fn)
    
    def run(self,
            initialInstance: nx.Graph,
            max_iter: int, 
            temperature: int,
            constant: int):
        return super().run(
            initialInstance,
            max_iter,
            lambda _: temperature,
            constant
        )
