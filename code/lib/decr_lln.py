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
        self.ES_M_s = [] 
        self.p_s = [] 
        self.ES_S_s = [] 

        print("Making ES_M_s")
        for i in tqdm(range(self.I_range)):
            ES_S_i = dict() 
            S_i = set()
            q_i = min(1, c*log(self.n)/(self.eps*2**i))
            # print("PROBABILITY: ", q_i)
            p_s_i = dict()
            for v in self.base_graph.nodes:
                if get_cointoss(q_i):
                    S_i.add(v) 
            ES_M_i = ESAlgov2(self.base_graph,S_i)
            for v in self.base_graph.nodes:
                if v in ES_M_i.es_graph.nodes:  
                    p_s_i[v] = ES_M_i.get_root(v)
            self.S_s.append(S_i)
            self.q_s.append(q_i)
            self.ES_M_s.append(ES_M_i)
            self.p_s.append(p_s_i)
        


        #Multi-threaded version
        if (MULTI_THREAD):
            print("Making ES_S_s")
            with mp_pool.ThreadPool(N_THREDS) as pool:
                for i in (range(self.I_range)):
                    ES_S_i = dict()
                    for w in (self.S_s[i]):
                        ES_S_i[w] = pool.apply_async(ESAlgov2, args=(self.base_graph,w,2**(i+2)))        
                    self.ES_S_s.append(ES_S_i)

                for i in (range(self.I_range)):
                    print(f"Waiting for {i}")
                    ES_S_i = self.ES_S_s[i]
                    for w in tqdm(self.S_s[i]):
                        ES_S_i[w] = ES_S_i[w].get()

        #Single-threaded version
        else:
            print("Making ES_S_s")
            
            for i in (range(self.I_range)):
                ES_S_i = dict()
                for w in tqdm(self.S_s[i]):
                    ES_S_i[w] = ESAlgov2(self.base_graph,w,2**(i+2))
                self.ES_S_s.append(ES_S_i)

            

    def delete(self, u:int, v:int):
        assert (u,v) in self.base_graph.edges
        self.base_graph.remove_edge(u,v)
        for i in range(self.I_range):
            ES_M_i:ESAlgov2 = self.ES_M_s[i]
            if((u,v) in ES_M_i.es_graph.edges): 
                ES_M_i.es_delete_oneshot(u,v)
                shifted_nodes = ES_M_i.shifted 
                for node in shifted_nodes:
                    if node in ES_M_i.es_graph.nodes:
                        self.p_s[i][node] = ES_M_i.get_root(node) 
                    else:
                        self.p_s[i].pop(node)

                for w in self.S_s[i]:
                    ES_S_i_w:ESAlgov2 = self.ES_S_s[i][w] 
                    if((u,v) in ES_S_i_w.es_graph.edges):
                        ES_S_i_w.es_delete_oneshot(u,v)
    
    def query_linear(self,u:int,v:int):
        for i in range(self.I_range):
            if u in self.p_s[i].keys():
                w = self.p_s[i][u] 
                ES_S_i_w = self.ES_S_s[i][w]
                if u in ES_S_i_w.es_graph.nodes and v in ES_S_i_w.es_graph.nodes: 
                    return ES_S_i_w.get_level(u) + ES_S_i_w.get_level(v) 
        return float("inf") 
    
    def binsearch_eval(self,u,v,i):
        if not u in self.p_s[i]: 
            status = STATUS_HIGH
        else: 
            w = self.p_s[i][u]
            if not v in self.ES_S_s[i][w].es_graph.nodes:
                status = STATUS_LOW 
            else:
                status = STATUS_OK
        if status == STATUS_OK: 
            ES_S_i_w:ESAlgov2 = self.ES_S_s[i][w]
            return (status, ES_S_i_w.get_level(u) + ES_S_i_w.get_level(v))
        else:
            return (status, float("inf"))
        
    def query_binsearch(self,u:int,v:int):
        i_L, i_H = 0, self.I_range - 1
        status_L, val_L = self.binsearch_eval(u,v,i_L) 
        status_H, val_H = self.binsearch_eval(u,v,i_H)
        if status_L == STATUS_OK:
            return val_L
        assert status_L == STATUS_LOW and status_H == STATUS_HIGH

        while i_H - i_L > 1:
            i_mid = (i_L + i_H)//2 
            status_mid, val_mid = self.binsearch_eval(u,v,i_mid)
            if status_mid == STATUS_LOW: 
                i_L = i_mid 
                status_L = status_mid
                val_L = val_mid
            else:
                if status_H == STATUS_OK:
                    assert status_mid == STATUS_OK 
                i_H = i_mid
                status_H = status_mid
                val_H = val_mid

                if(status_mid == STATUS_OK):
                    break
        if status_mid == STATUS_OK:
            return val_mid
        else:
            return float("inf")
        
if __name__ == "__main__":
    MULTI_THREAD = True
    N_THREDS = 11
    prob = 1e-6
    while True:
        try:
            base_graph = get_connected_gnp_graph(16000,8000,prob)
            break
        except AssertionError:
            prob *= 2
            print(f"\rProbability: {prob}, {" "*5}", end="")
    c = 0.01
    epsilon = 0.5
    print("\n\nBase graph nodes and edges: ", len(base_graph.nodes), len(base_graph.edges))
    decr_algo = DecrAPSPAlgo(base_graph,epsilon,c)

    while True:
        total = 0
        fails = 0
        uv_s = list(decr_algo.base_graph.edges)
        if(len(uv_s) < 100):
            break
        for _ in range(random.randint(10,100)):
            uv = random.choice(uv_s)
            uv_s.remove(uv)
            u,v = uv
            decr_algo.delete(u,v)
        for u,v in tqdm(list(combinations(decr_algo.base_graph.nodes,2))):
            try: 
                true_l = nx.shortest_path_length(decr_algo.base_graph,u,v)
                if not decr_algo.query_linear(u,v) <= true_l*(1+epsilon):
                    fails += 1
                total += 1
            except nx.NetworkXNoPath:
                continue
        print(f"Total: {total}, Fails: {fails}")



                

