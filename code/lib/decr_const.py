from itertools import combinations, product
from math import floor, isfinite, log
from graphfig import * 
from bfs import *
from even_shiloach import *
from oracle import * 
from decr_lln import *
import networkx as nx
from collections import deque
import multiprocessing as mp 
import multiprocessing.pool as mp_pool


class DecrAPSPConstTAlgo(StatAlgo):

    def __init__(self,base_graph, epsilon:float, c:float):
        super().__init__(base_graph,copy=False)
        self.epsilon = epsilon
        self.c = c
        print("Making decr lln.")
        self.decr_lln = DecrAPSPAlgo(self.base_graph,self.epsilon,self.c)
        self.n = len(self.base_graph.nodes)
        print("Making oracle.")
        self.oracle:Oracle = Oracle(base_graph,k=2,d=self.n**0.5)
        self.r = floor(0.5*log(self.n,2))
        print("R: ", self.r)
        self.S_r = self.decr_lln.S_lst[self.r]
        print("Len(S_r): ", len(self.S_r))
        self.ES_M_r = self.decr_lln.S_trees[self.r]
        self.dist_pairs_r = dict()
        print("Making dist_pairs_r.")
        for (u,v) in tqdm(list(combinations(self.S_r,2))):
            u,v = min(u,v),max(u,v) 
            self.dist_pairs_r[(u,v)] = self.decr_lln.query_linear(u,v)
        for u in self.S_r:
            self.dist_pairs_r[(u,u)] = 0
        
    def query(self,u,v):
        oracle_guess = self.oracle.double_distance_query(u,v)

        
        if oracle_guess < 3*self.n**0.5:
            i_start = max(floor(log(oracle_guess/3,2)),0)
            lln_guesses = [self.decr_lln.evaluate_S_i(u,v,i)[1] for i in range(i_start,i_start + 3) if i < self.decr_lln.I_range]
            return min(lln_guesses + [oracle_guess,])
        else: 
            a = self.ES_M_r.get_root(u)
            b = self.ES_M_r.get_root(v) 
            if a > b:
                a,b = b,a
            return self.dist_pairs_r[(a,b)] + self.ES_M_r.get_level(u) + self.ES_M_r.get_level(v)
    def recompute_dist_pairs_r(self):
        self.dist_pairs_r = dict()
        for (u,v) in list(combinations(self.S_r,2)):
            u,v = min(u,v),max(u,v) 
            self.dist_pairs_r[(u,v)] = self.decr_lln.query_linear(u,v)

    def delete(self,u,v):
        self.oracle.delete(u,v)
        self.decr_lln.delete(u,v)
        if(u in self.S_r and v in self.S_r):
            self.dist_pairs_r.pop((min(u,v),max(u,v)))
        self.base_graph.remove_edge(u,v)
        
if __name__ == "__main__":
    
    prob = 1e-6
    while True:
        try:
            base_graph = get_connected_gnp_graph(1200,800,prob)
            break
        except AssertionError:
            prob *= 2
            # print(f"\rProbability: {prob}, {" "*5}", end="")
    c = 0.01
    epsilon = 0.5
    print("\n\nBase graph nodes and edges: ", len(base_graph.nodes), len(base_graph.edges))
    decr_algo = DecrAPSPConstTAlgo(base_graph,epsilon,c)

    first_time = True
    while True:
        total = 0
        fails = 0
        if(not first_time):
            uv_s = list(decr_algo.base_graph.edges)
            if(len(uv_s) < 100):
                break
            for _ in range(random.randint(10,100)):
                uv = random.choice(uv_s)
                uv_s.remove(uv)
                u,v = uv
                decr_algo.delete(u,v)
        else:
            first_time = False
        
        for u,v in (list(combinations(decr_algo.base_graph.nodes,2))):
            try: 
                true_uv_dist = nx.shortest_path_length(decr_algo.base_graph,u,v)


                if not decr_algo.query(u,v) <= true_uv_dist*(1+epsilon):
                    fails += 1
                total += 1
            except nx.NetworkXNoPath:
                continue
        print(f"Total: {total}, Fails: {fails}")