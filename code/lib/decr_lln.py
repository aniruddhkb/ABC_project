from itertools import combinations, product
from math import floor, isfinite, log
from graphfig import * 
from bfs import *
from even_shiloach import *
import networkx as nx
from collections import deque
import multiprocessing as mp 
import multiprocessing.pool as mp_pool
from pprint import pprint
import random 
random.seed(29824)

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
        self.S_lst = []
        self.q_lst = []
        self.S_trees: list[ESAlgov2] = [] 
        self.S_parent = [] 
        self.V_trees: list[dict[int,ESAlgov2]] = [] 

        print("Making S_lst.")
        for i in (range(self.I_range)):
            
            S_i = []
            q_i = min(1, c*log(self.n)/(self.eps*2**i))
            for v in self.base_graph.nodes:
                if get_cointoss(q_i):
                    S_i.append(v) 
            self.S_lst.append(S_i)
        print(f"Made S_lst. Lens = {[len(S_i) for S_i in self.S_lst]}" )
        # print(self.S_lst)
        print("Making S_trees.")
        for i in tqdm(range(self.I_range)):
            
            S_i = self.S_lst[i] 
            S_tree_i = ESAlgov2(self.base_graph,S_i)
            for v in self.base_graph.nodes:
                try:
                    assert v in S_tree_i.es_graph.nodes
                except AssertionError:  
                    print(f"{v} not in {S_tree_i.es_graph.nodes}")
                    raise AssertionError
                try: 
                    assert S_tree_i.get_level(v) <= self.eps * 2**i
                except AssertionError:
                    print(f"Level of node {v} in tree S_tree {i} = {S_tree_i.get_level(v)} > {self.eps * 2**i}")
                    raise AssertionError

            self.S_lst.append(S_i)
            self.q_lst.append(q_i)
            self.S_trees.append(S_tree_i)
        print("Made S_trees.")
            
        


        #Multi-threaded version
        if (MULTI_THREAD):
            
            print("MULTI THREAD.")
            print("Making V_trees")
            with mp_pool.ThreadPool(N_THREADS) as pool:
                for i in (range(self.I_range)):
                    V_trees_i = dict()
                    for w in (self.S_lst[i]):
                        V_trees_i[w] = pool.apply_async(ESAlgov2, args=(self.base_graph,w,2**(i+2)))        
                    self.V_trees.append(V_trees_i)

                for i in (range(self.I_range)):
                    print(f"Waiting for {i}")
                    V_trees_i = self.V_trees[i]
                    for w in tqdm(self.S_lst[i]):
                        V_trees_i[w] = V_trees_i[w].get()

        #Single-threaded version
        else:
            print("Single threaded; making V-trees.")
            
            for i in tqdm(range(self.I_range)):
                V_trees_i = dict()
                for w in (self.S_lst[i]):
                    V_trees_i[w] = ESAlgov2(self.base_graph,w,2**(i+2))
                self.V_trees.append(V_trees_i)
        
        print("Getting uv distances.")

        uv_dists = dict(nx.all_pairs_shortest_path_length(self.base_graph))   

        print("Checking distances.")
        for u,v in tqdm(list(combinations(self.base_graph.nodes,2))):
            uv_dist = uv_dists[u][v]
            for i in range(self.I_range):
                    if uv_dist <= self.eps*2**(i+1):
                        w = self.S_trees[i].get_root(u)
                        w_V_tree_i = self.V_trees[i][w]
                        try:
                            assert v in w_V_tree_i.es_graph.nodes
                        except AssertionError:
                            print(f"Node {v} not in V-tree {i} of S_i-parent of {u} even though distance is {uv_dist} <= {self.eps*2**(i+1)}")
                            raise AssertionError

    def delete(self, u:int, v:int):
        assert (u,v) in self.base_graph.edges
        self.base_graph.remove_edge(u,v)
        for i in range(self.I_range):
            S_tree_i:ESAlgov2 = self.S_trees[i]
            if((u,v) in S_tree_i.es_graph.edges): 
                S_tree_i.es_delete_oneshot(u,v)

            for w in self.S_lst[i]:
                V_tree_i_w:ESAlgov2 = self.V_trees[i][w] 
                if((u,v) in V_tree_i_w.es_graph.edges):
                    V_tree_i_w.es_delete_oneshot(u,v)
    
    def query_linear(self,u:int,v:int):
        to_return_ans = float("inf")
        for i in range(self.I_range):
            ans = self.evaluate_S_i(u,v,i)
            if ans < to_return_ans:
                to_return_ans = ans
        return to_return_ans

    def evaluate_S_i(self,u,v,i):
        S_tree_i:ESAlgov2 = self.S_trees[i]
        if u in S_tree_i.es_graph.nodes:
            w = S_tree_i.get_root(u)
            assert S_tree_i.get_level(u) <= self.eps*2**i

            V_tree_i_w:ESAlgov2 = self.V_trees[i][w]
            assert u in V_tree_i_w.es_graph.nodes
            assert w in V_tree_i_w.multi_roots

            if v in V_tree_i_w.es_graph.nodes:
                return V_tree_i_w.get_level(u) + V_tree_i_w.get_level(v)
            else:
                return float("inf")
            
    def query_binsearch(self,u,v): 
        low_idx = 0
        high_idx = self.I_range - 1
        low_ans = self.evaluate_S_i(u,v,low_idx)
        low_status = isfinite(low_ans)
        
        if(low_status): 
            return low_ans 
        else: 
            high_ans = self.evaluate_S_i(u,v,high_idx)
            high_status = isfinite(high_ans)
            if not high_status:
                return float("inf")
        while high_idx - low_idx > 1:
            mid_idx = (high_idx + low_idx)//2
            mid_ans = self.evaluate_S_i(u,v,mid_idx)
            mid_status = isfinite(mid_ans)
            if mid_status:
                high_idx = mid_idx
                high_ans = mid_ans
                high_status = mid_status
            else:
                low_idx = mid_idx
                low_ans = mid_ans
                low_status = mid_status
        return high_ans
            
    
        
if __name__ == "__main__":
    try:
        import syscheck
        syscheck.syscheck()
        MULTI_THREAD = True
    except RuntimeError:
        MULTI_THREAD = False
    N_THREADS = 11
    prob = 0.005
    while True:
        try:
            base_graph: nx.Graph = get_connected_gnp_graph(900,600,prob)
            break
        except AssertionError:
            prob *= 2
    print("GNP PROB: ", prob)
    c = 0.5
    epsilon = 0.5
    print(f"Base graph nodes and edges: {base_graph.number_of_nodes()}, {base_graph.number_of_edges()} ")

    decr_algo = DecrAPSPAlgo(base_graph,epsilon,c)

    print("TESTING.")
    while True:
        total = 0
        fails = 0
        fails_type_2 = 0
        uv_s = list(decr_algo.base_graph.edges)
        if(len(uv_s) < 100):
            break
        print("DELETING EDGES.")
        n_deletions = 0
        for _ in tqdm(range(random.randint(50,100))):
            uv = random.choice(uv_s)
            base_graph.remove_edge(*uv)
            if not nx.is_connected(base_graph):
                base_graph.add_edge(*uv)
                continue
            n_deletions += 1
            uv_s.remove(uv)
            u,v = uv
            decr_algo.delete(u,v)
        print(f"Deleted {n_deletions} edges.")
        true_uv_dists = dict(nx.all_pairs_shortest_path_length(decr_algo.base_graph))
        for u,v in tqdm(list(combinations(decr_algo.base_graph.nodes,2))):
            try: 
                true_ans = true_uv_dists[u][v]
            except KeyError or nx.NetworkXNoPath:
                continue
            given_ans = decr_algo.query_linear(u,v)
            if not (true_ans <= given_ans and given_ans <= (1+epsilon)*true_ans):

                print(f"FAIL: {u} {v}; T = {true_ans}, Pred:{given_ans}")
                fails += 1
            # if not decr_algo.query_binsearch(u,v) == given_ans:
                # fails_type_2 += 1
            total += 1
            
        # print(f"Total: {total}, Fails: {fails}, Fails type 2: {fails_type_2}")
        print(f"Total: {total}, Fails: {fails}")



                

