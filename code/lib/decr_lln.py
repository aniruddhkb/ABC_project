from itertools import combinations, product
from math import floor, log
from graphfig import * 
from bfs import *
from even_shiloach import *
import networkx as nx
from collections import deque
import multiprocessing as mp 
import multiprocessing.pool as mp_pool

import random 
random.seed(31415)

STATUS_LOW = -1
STATUS_OK = 0 
STATUS_HIGH = 1
MULTI_THREAD = False
get_cointoss = lambda p: random.binomialvariate(1,p)



class DecrAPSPAlgo(DynAlgo):

    

    def __init__(self, base_graph:nx.Graph,eps:float,c:float):
        DynAlgo.__init__(self, base_graph)
        self.eps = eps
        self.c = c 
        self.n = len(self.base_graph.nodes) 
        self.I_range = floor(log(self.n,2)) + 1 
        self.S_s = []
        self.q_s = []
        self.ES_M = [] 
        self.p_s = [] 
        self.ES_T = [] 

        # print("Making ES_M_s")
        for i in (range(self.I_range)):
            ES_T_i = dict() 
            S_i = set()
            q_i = min(1, c*log(self.n)/(self.eps*2**i))
            # # print("PROBABILITY: ", q_i)
            for v in self.base_graph.nodes:
                if get_cointoss(q_i):
                    S_i.add(v) 
            ES_M_i = ESAlgov2(self.base_graph,S_i)
            self.S_s.append(S_i)
            self.q_s.append(q_i)
            self.ES_M.append(ES_M_i)
            
        


        #Multi-threaded version
        if (MULTI_THREAD):
            # print("Making ES_S_s")
            with mp_pool.ThreadPool(N_THREDS) as pool:
                for i in (range(self.I_range)):
                    ES_T_i = dict()
                    for w in (self.S_s[i]):
                        ES_T_i[w] = pool.apply_async(ESAlgov2, args=(self.base_graph,w,2**(i+2)))        
                    self.ES_T.append(ES_T_i)

                for i in (range(self.I_range)):
                    # print(f"Waiting for {i}")
                    ES_T_i = self.ES_T[i]
                    for w in (self.S_s[i]):
                        ES_T_i[w] = ES_T_i[w].get()

        #Single-threaded version
        else:
            # print("Making ES_S_s")
            
            for i in (range(self.I_range)):
                ES_T_i = dict()
                for w in (self.S_s[i]):
                    ES_T_i[w] = ESAlgov2(self.base_graph,w,2**(i+2))
                self.ES_T.append(ES_T_i)

    def delete(self, u:int, v:int):
        assert (u,v) in self.base_graph.edges
        self.base_graph.remove_edge(u,v)
        for i in range(self.I_range):
            ES_M_i:ESAlgov2 = self.ES_M[i]
            if((u,v) in ES_M_i.es_graph.edges): 
                ES_M_i.es_delete_oneshot(u,v)

            for w in self.S_s[i]:
                ES_T_i_w:ESAlgov2 = self.ES_T[i][w] 
                if((u,v) in ES_T_i_w.es_graph.edges):
                    ES_T_i_w.es_delete_oneshot(u,v)
    
    def query_linear(self,u:int,v:int):
        for i in range(self.I_range):
            M_i:ESAlgov2 = self.ES_M[i]
            if u in M_i.es_graph.nodes:
                w = M_i.get_root(u)
                ES_T_i_w:ESAlgov2 = self.ES_T[i][w]
                if u in ES_T_i_w.es_graph.nodes and v in ES_T_i_w.es_graph.nodes: 
                    return ES_T_i_w.get_level(u) + ES_T_i_w.get_level(v) 
        return float("inf")

    def evaluate_S_i(self,u,v,i):
        if not u in self.ES_M[i].es_graph.nodes: 
            status = STATUS_HIGH
        else: 
            w = self.ES_M[i].get_root(u)
            ES_T_i_w_nodes =  self.ES_T[i][w].es_graph.nodes
            if not v in ES_T_i_w_nodes or not u in ES_T_i_w_nodes:
                status = STATUS_LOW 
            else:
                status = STATUS_OK
        if status == STATUS_OK: 
            ES_T_i_w:ESAlgov2 = self.ES_T[i][w]
            return (status, ES_T_i_w.get_level(u) + ES_T_i_w.get_level(v))
        else:
            return (status, float("inf"))
        
    
        
if __name__ == "__main__":
    MULTI_THREAD = False
    N_THREDS = 11
    prob = 1e-6
    while True:
        try:
            base_graph = get_connected_gnp_graph(1200,800,prob)
            break
        except AssertionError:
            prob *= 2
            
    c = 0.01
    epsilon = 0.5
    
    decr_algo = DecrAPSPAlgo(base_graph,epsilon,c)

    while True:
        total = 0
        fails = 0
        fails_type_2 = 0
        uv_s = list(decr_algo.base_graph.edges)
        if(len(uv_s) < 100):
            break
        for _ in range(random.randint(10,100)):
            uv = random.choice(uv_s)
            uv_s.remove(uv)
            u,v = uv
            decr_algo.delete(u,v)
        for u,v in (list(combinations(decr_algo.base_graph.nodes,2))):
            try: 
                true_l = nx.shortest_path_length(decr_algo.base_graph,u,v)


                if not decr_algo.query_linear(u,v) <= true_l*(1+epsilon):
                    fails += 1
                if not decr_algo.query_binsearch(u,v) == decr_algo.query_linear(u,v):
                    fails_type_2 += 1
                total += 1
            except nx.NetworkXNoPath:
                continue
        # print(f"Total: {total}, Fails: {fails}, fails_type_2: {fails_type_2}")



                

