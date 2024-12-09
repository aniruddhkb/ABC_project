from itertools import combinations, product
from math import floor
from graphfig import * 
from bfs import *
from even_shiloach import *
import networkx as nx
from collections import deque
import random 
random.seed(2412)

get_cointoss = lambda p: random.binomialvariate(1,p)

class Oracle(DynAlgo):

    def __init__(self, base_graph:nx.Graph, k:int, d:int): 
        DynAlgo.__init__(self, base_graph) 
        self.oracle_graph = self.base_graph
        self.n = len(self.oracle_graph.nodes)
        self.all_graphs["oracle_base"] = self.oracle_graph

        self.k = k
        self.d = d
        self.dbar = (2*self.k - 1)*d 

        self.A_s = []
        self.A_s.append(set(self.oracle_graph.nodes))
        
        self.cointoss_prob = self.n**(-1/self.k)
        for i in range(1,self.k):
            A_i = set()
            for node in self.A_s[-1]:
                if get_cointoss(self.cointoss_prob):
                    A_i.add(node)
            self.A_s.append(A_i)
        self.A_s.append(set())
        self.A_bar_s = []

        self.A_trees:list[ESAlgov2] = list()

        for i in range(self.k):
            self.A_bar_s.append(self.A_s[i] - self.A_s[i+1])
        for i in range(self.k):
            self.A_trees.append(ESAlgov2(self.oracle_graph,self.A_s[i],self.dbar))
        
        self.V_trees:dict[int,ESAlgov2] = dict()
        for v in self.oracle_graph.nodes:
            self.V_trees[v] = ESAlgov2(self.oracle_graph,v,self.dbar)

    def double_distance_query(self,u:int,v:int)->int|float:
        return min(
            self.distance_query(u,v),
            self.distance_query(v,u)
        )
    def distance_query(self,u:int,v:int)->int|float:
        assert u in self.oracle_graph.nodes and v in self.oracle_graph.nodes
        
        w = u 
        i = 0
        while True:
            if i == self.k - 1 or self.V_trees[w].get_level(v) < self.A_trees[i + 1].get_level(v):
                break 
            i += 1
            u, v = v, u
            
            try:
                w = self.A_trees[i].get_root(u)
            except :
                raise KeyError
            assert self.A_trees[i].get_level(u) == self.V_trees[w].get_level(u)
        
        return self.V_trees[w].get_level(v) + self.V_trees[w].get_level(u) 
        
            
    def delete(self,u:int,v:int)->None: 
        assert u in self.oracle_graph.nodes and v in self.oracle_graph.nodes

        for w in self.oracle_graph.nodes:
            V_tree_w:ESAlgov2 =  self.V_trees[w]
            if((u,v) in V_tree_w.es_graph.edges):
                V_tree_w.es_delete_oneshot(u,v)
        
        for i in range(self.k):
            A_tree_i:ESAlgov2 = self.A_trees[i]
            if((u,v) in A_tree_i.es_graph.edges):
                A_tree_i.es_delete_oneshot(u,v)
        
        self.oracle_graph.remove_edge(u,v)
                    

if __name__ == "__main__":
    
    prob = 5e-3
    while True:
        try:
            base_graph: nx.Graph = get_connected_gnp_graph(900,600,prob)
            break
        except AssertionError:
            prob *= 2
    k= 2 
    d = floor(base_graph.number_of_nodes()**0.5)
    
    eps = 2*k - 1 

    oracle = Oracle(base_graph,k,d) 
    while True:
        total = 0
        fails = 0
        for u, v in tqdm(list(combinations(oracle.oracle_graph.nodes,2))):
            true_ans = nx.shortest_path_length(oracle.oracle_graph,u,v)
            if true_ans <= d:
                total += 1
                given_ans = oracle.double_distance_query(u,v)
                # if(given_ans == float("inf")):
                
                if(not (true_ans <= given_ans and given_ans <= eps*true_ans)):
                    print(f"FAIL: {u} {v}; T = {true_ans}, Pred:{given_ans}")
                    fails += 1
            
        print(f"\nTotal: {total}, Fails: {fails}")    
        print("\n")

        uv_s = list(oracle.base_graph.edges)
        if(len(uv_s) < 100):
            break
        print("DELETING EDGES.")
        n_deletions = 0
        for _ in tqdm(range(random.randint(50,100))):
            skip = False
            uv = random.choice(uv_s)
            oracle.base_graph.remove_edge(*uv)
            if not nx.is_connected(oracle.base_graph):
                skip  = True
            oracle.base_graph.add_edge(*uv)
            if(skip):
                continue
            uv_s.remove(uv)
            
            oracle.delete(*uv)
            n_deletions += 1
        print(f"Deleted {n_deletions} edges.")
                
        
                                
                        
                
            
            
            

                    



