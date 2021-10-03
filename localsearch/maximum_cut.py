# %% 
import networkx as nx
from localsearch.local_search import NeighborhoodRelation
from typing import FrozenSet, List, Tuple, Optional
from abc import ABC
from itertools import combinations

class MaximumCutRelation(NeighborhoodRelation, ABC):
    def __init__(self, p1_symbol: str, p2_symbol: str) -> None:
        super().__init__(nx.Graph)
        self.p1_symbol = p1_symbol
        self.p2_symbol = p2_symbol
    
    def partition_symbols(self):
        return self.p1_symbol, self.p2_symbol



class SingleFlipRelation(MaximumCutRelation):
    """ 
        The Single-Flip-Relation from section 4.1. Generates new instances by moving
        one vertex A->B or B->A with A != {} and B != {}.

        NOTE: Implemented through KFlipRelation with k=1. It is redundant,
        but left here for clarity and conformness with the paper of the project. 
    """

    def __init__(self, p1_symbol: str, p2_symbol: str) -> None:
        super().__init__(p1_symbol, p2_symbol)
        self.kflip = KFlipRelation(1, p1_symbol, p2_symbol)

    def neighbors(self, instance: nx.Graph, exclude_marked: Optional[bool] = False) -> FrozenSet[nx.Graph]:
        return self.kflip.neighbors(instance, exclude_marked)

class KFlipRelation(MaximumCutRelation):
    """ 
        The K-Flip-Relation from section 4.2. Generates new instances by moving
        at most K vertices A->B or B->A with A != {} and B != {}.
    """

    def __init__(self, k: int, p1_symbol: str, p2_symbol: str) -> None:
        super().__init__(p1_symbol, p2_symbol)
        self.k = k
        self.excluded = []

    def neighbors(self, instance: nx.Graph, exclude_marked: bool = False) -> FrozenSet[nx.Graph]:

        p1 = MaximumCut.split_partition(instance, self.p1_symbol, exclude_marked)
        p2 = MaximumCut.split_partition(instance, self.p2_symbol, exclude_marked)

        neighborhood = []
        for k in range(self.k):
            neighborhood.extend(self.flip(p1, p2, k + 1))
        
        graph_neighborhood = []
        for solution in neighborhood:
            graph_neighborhood.append(self._partition_graph(instance, solution))
        
        return graph_neighborhood

    def flip(self, partition1: List[str], partition2: List[str], k: int) -> List[Tuple[List[str], List[str]]]:
        flipped = []
        flipped.extend(self.half_flip(partition1, partition2, k))
        flipped.extend(self.half_flip(partition2, partition1, k, reverse=True))
        return flipped
    
    def half_flip(self,
                  delete: List[str],
                  append: List[str], k: int,
                  reverse: bool = False,) -> List[Optional[Tuple[List[str], List[str]]]]:
        half_flipped = []
        if len(delete) > k:
            p1_powerset = combinations(delete, k)

            for subset in p1_powerset:
                new_part1 = [node for node in delete if node not in set(subset)]
                new_part2 = append.copy()
                new_part2.extend(list(subset))

                if not reverse:
                    half_flipped.append((
                        new_part1,
                        new_part2
                    ))
                else:
                    half_flipped.append((
                        new_part2,
                        new_part1
                    ))

        return half_flipped
    
    def _partition_graph(self, graph: nx.Graph, mapping) -> List[nx.Graph]:
        partitioned_graph = graph.copy()
        for node in partitioned_graph.nodes:
            partitioned_graph.nodes[node]['partition'] = self.p1_symbol if node in mapping[0] else self.p2_symbol
        return partitioned_graph

# %%
class KernighanLinHeuristic(MaximumCutRelation):

    def __init__(self, p1_symbol: str, p2_symbol: str, cost_fn) -> None:
        super().__init__(p1_symbol, p2_symbol)
        self.single_flip = SingleFlipRelation(self.p1_symbol, self.p2_symbol)
        self.cost_fn = cost_fn

    def neighbors(self, instance: nx.Graph) -> FrozenSet[nx.Graph]:
        for node in instance.nodes:
            instance.nodes[node]['marked'] = False
        
        neighborhood = []
        current_instance = instance
        while True:
            sf_neighborhood = self.single_flip.neighbors(current_instance, exclude_marked=True)
            max_val, max_sol = -1, None
            for solution in sf_neighborhood:
                val = self.cost_fn(solution, self.p1_symbol, self.p2_symbol)
                if val > max_val:
                    max_val, max_sol = val, solution
            neighborhood.append(max_sol)
            current_instance = self._mark_differing_nodes(current_instance, max_sol)
            marked_values = [current_instance.nodes[node]['marked'] for node in current_instance.nodes]
            if sum(marked_values) == len(current_instance.nodes) - 2:
                break
        return neighborhood
        
    def _mark_differing_nodes(self, original: nx.Graph, differing: nx.Graph):
        for orig, diff in zip(
                        sorted(original.nodes),
                        sorted(differing.nodes)
                       ):
            if original.nodes[orig]['partition'] != differing.nodes[diff]['partition']:
                differing.nodes[diff]['marked'] = True
        return differing

#%%
class MaximumCut:
    @staticmethod
    def split_partition(graph: nx.Graph, pname: str, exclude_marked: bool) -> List[str]:
        return [node for (node, data) in graph.nodes(data=True)\
             if data['partition'] == pname and not (exclude_marked and data['marked'])]
    
    @staticmethod
    def cost(graph: nx.Graph, p1symbol: str, p2symbol: str) -> float:
        p1 = MaximumCut.split_partition(graph, p1symbol, False)
        p2 = MaximumCut.split_partition(graph, p2symbol, False)

        weight_total = 0
        for v1, v2, data in graph.edges(data=True):
            if (v1 in p1 and v2 in p2) or \
                (v1 in p2 and v2 in p1):
                weight_total += data["weight"]
        
        return float(weight_total)
