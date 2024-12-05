from graphfig import * 
from bfs import * 


class FastSparseCovers(StatAlgo):
    def __init__(self, base_graph:nx.Graph, r:int, beta:float):
        super().__init__(base_graph) 
        self.r = r
        self.beta = beta

    def grow_one_cluster(self,R:set[int], U:set[int], seed_vert:int, Y:list[set[int]]): 
        phi_z = set()
        phi_z.add(seed_vert) 

        G2 = self.base_graph.subgraph()