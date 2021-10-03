from typing import FrozenSet, Callable
from abc import ABC, abstractmethod

class NeighborhoodRelation(ABC):
    def __init__(self, instanceType) -> None:
        self.instanceType = instanceType

    @property
    def type(self):
        return self.instanceType

    @abstractmethod
    def neighbors(self, instance) -> FrozenSet[object]:
        ...


class LocalSearchAlgorithm(ABC):
    def __init__(self, relation: NeighborhoodRelation, cost_fn: Callable):
        self.relation = relation
        self.cost = cost_fn
        self.p1_symbol = self.relation.partition_symbols()[0]
        self.p2_symbol = self.relation.partition_symbols()[1]

    @abstractmethod
    def run(self, initialInstance, max_iter: int):
        ...
