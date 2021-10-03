# %%
from typing import Callable, Optional
import networkx as nx
import random
from tqdm import tqdm
import os
import csv
import shutil

class Experiment:
    def __init__(
        self,
        name: str,
        cost: Callable,
        relations: dict,
        algorithms: dict,
        num_instances: int,
        node_count: int,
        prob_edge_spawn: float
    ) -> None:
        random.seed(1)
        self.name = name

        self.p1, self.p2 = 'A', 'B'

        self.cost = cost
        self.relations = self._generate_relations(relations)
        self.algorithms = algorithms

        self.num_instances = num_instances
        self.node_count = node_count
        self.prob_edge_spawn = prob_edge_spawn

    def start(self, dirname: Optional[str] = None):
        graphs = []
        for _ in tqdm(range(self.num_instances), desc='Graph generation'):
            g = nx.gnp_random_graph(self.node_count, self.prob_edge_spawn)

            for n in g.nodes():
                p = self.p1 if random.random() < 0.5 else self.p2
                g.nodes[n]["partition"] = p

            for e in g.edges():
                weight = random.random()
                g.edges[e]["weight"] = weight

            graphs.append(g)
        
        results = {alg: {} for alg in self.algorithms}

        for alg, cls_info in zip(self.algorithms, self.algorithms.values()):
            for rel in tqdm(self.relations.values(), desc=alg):
                results[alg][rel] = []
                search = cls_info[0](relation=rel, cost_fn=self.cost)
                for g in graphs:
                    sol = search.run(initialInstance=g, **cls_info[1])
                    results[alg][rel].append((sol, self.cost(sol, self.p1, self.p2)))

        self._display_results(results, self.algorithms)
        if dirname is not None:
            self._write_results_file(dirname, results)
    
    def _display_results(self, results: dict, info: dict):
        print("=================================")
        print(self.name)
        for alg, res_dict in results.items():
            print("=================================")
            print(f"Algorithm {alg}:")
            alg_info = info[alg][1]
            for att_name, val in alg_info.items():
                print(f"{att_name}: {val}")
            for rel, res in res_dict.items():
                costs = [c for _, c in res]
                print(f"Average cost ({rel.__class__.__name__}): {sum(costs)/len(costs)}")
            print("\n\n")
    
    def _write_results_file(self, dirname: str, results: dict):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        for alg, res_dict in results.items():
            formatted_dict = {}
            for rel, res in res_dict.items():
                costs = [c for _, c in res]
                name = rel.__class__.__name__
                suffix = 1
                while name in formatted_dict.keys():
                    name += str(suffix)
                    suffix += 1
                formatted_dict[name] = sum(costs)/len(costs)
            filepref = alg
            suffix = 1
            while os.path.exists(filepref + '.csv'):
                filepref += str(suffix)
                suffix += 1
            with open(os.path.join(dirname, filepref + '.csv'), 'w+') as f:
                writer = csv.writer(f)
                for row in formatted_dict.items():
                    writer.writerow(row)

    def _generate_relations(self, rels: dict):
        ret = {}
        for rname, info in zip(rels, rels.values()):
            ret[rname] = info[0](p1_symbol=self.p1, p2_symbol=self.p2, **info[1])
        return ret
