# %%
import os
from core.experiment import Experiment
from localsearch.maximum_cut import MaximumCut, SingleFlipRelation, KFlipRelation, KernighanLinHeuristic
from localsearch.gradient_ascent import GradientAscent
from localsearch.metropolis import Metropolis
from localsearch.simulated_annealing import SimulatedAnnealing
from localsearch.tabu_search import TabuSearch 
# %%
def main():
    min_node_count = 5
    max_node_count = 10
    base_dir = "default_paper"

    for node_count in range(min_node_count, max_node_count+1):
        Experiment(
            name=f"MaximumCut - {node_count}",
            cost=MaximumCut.cost,
            relations={
                'Single Flip': (SingleFlipRelation, {}),
                'KFlip (k=2)': (KFlipRelation, {'k': 2}),
                'KFlip (k=3)': (KFlipRelation, {'k': 3}),
                'KFlip (k=4)': (KFlipRelation, {'k': 4}),
                'Kernighan-Lin-Heuristic': (KernighanLinHeuristic, {'cost_fn': MaximumCut.cost})
            },
            algorithms={
                'Gradient Ascent': (GradientAscent, {'max_iter': 300}),
                'Metropolis Algorithmus': (Metropolis, {'max_iter': 300, 'temperature': 0.5, 'constant': 1}),
                'Simulated Annealing': (SimulatedAnnealing, {'max_iter': 300, 'cooling_schedule': lambda i: (99_994e-5)**((i*(i-1))/2) * 2, 'constant': 1}),
                'Tabu Search': (TabuSearch, {'max_iter': 300})
            },
            num_instances=200,
            node_count=node_count,
            prob_edge_spawn=0.1
        ).start(dirname=os.path.join(base_dir, str(node_count)))

if __name__ == "__main__":
    main()
