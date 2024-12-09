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
        
    def query(self,u,v, test_val = None):
        oracle_guess = self.oracle.double_distance_query(u,v)

        
        if oracle_guess < 3*self.n**0.5:
            i_start = max(floor(log(oracle_guess/3,2)  -1 ),0)
            lln_guesses = [self.decr_lln.evaluate_S_i(u,v,i) for i in range(i_start,i_start + 3) if i < self.decr_lln.I_range]
            ans = min(lln_guesses + [oracle_guess,])
            if test_val is not None:
                if ans < test_val:
                    print(f"u: {u}, v: {v}, oracle_guess: {oracle_guess}, lln_guesses: {lln_guesses}, ans: {ans}")
                    print("")
        else: 
            print('miss.')
            a = self.ES_M_r.get_root(u)
            b = self.ES_M_r.get_root(v) 
            if a > b:
                a,b = b,a
            ans = self.dist_pairs_r[(a,b)] + self.ES_M_r.get_level(u) + self.ES_M_r.get_level(v)
            if test_val is not None:
                if ans < test_val:
                    print(f"u: {u}, v: {v}, oracle_guess: {oracle_guess}, ans: {ans}")
        return ans
    def recompute_dist_pairs_r(self):
        self.S_r = self.decr_lln.S_lst[self.r]
        self.dist_pairs_r = dict()
        for (u,v) in list(combinations(self.S_r,2)):
            u,v = min(u,v),max(u,v) 
            self.dist_pairs_r[(u,v)] = self.decr_lln.query_binsearch(u,v)

    def delete(self,u,v):
        self.oracle.delete(u,v)
        self.decr_lln.delete(u,v)
        self.base_graph.remove_edge(u,v)
        self.recompute_dist_pairs_r()
        
if __name__ == "__main__":
    try:
        import syscheck
        syscheck.syscheck()
        MULTI_THREAD = True
    except RuntimeError:
        MULTI_THREAD = False
    N_THREADS = 11
    prob = 1e-6
    while True:
        try:
            base_graph: nx.Graph = get_connected_gnp_graph(900,600,prob)
            break
        except AssertionError:
            prob *= 2
            
    c = 0.5
    epsilon = 0.5
    print(f"Base graph nodes and edges: {base_graph.number_of_nodes()}, {base_graph.number_of_edges()} ")

    decr_constTalgo = DecrAPSPConstTAlgo(base_graph,epsilon,c)

    print("TESTING.")
    first_time = True
    while True:
        total = 0
        fails = 0
        uv_s = list(decr_constTalgo.base_graph.edges)
        if(len(uv_s) < 100):
            break
        if not first_time:
            print("DELETING EDGES.")
            n_deletions = 0
            for _ in tqdm(range(random.randint(10,100))):
                skip = False
                uv = random.choice(uv_s)
                uv_s.remove(uv)
                decr_constTalgo.base_graph.remove_edge(*uv)
                if not nx.is_connected(decr_constTalgo.base_graph):
                    skip = True
                decr_constTalgo.base_graph.add_edge(*uv)
                if not skip:
                    decr_constTalgo.delete(*uv)
                    n_deletions += 1
            print(f"N_DELETIONS {n_deletions}")
        else:
            first_time = False
        true_uv_dists = dict(nx.all_pairs_shortest_path_length(decr_constTalgo.base_graph))
        for u,v in tqdm(list(combinations(decr_constTalgo.base_graph.nodes,2))):
            try: 
                true_ans = true_uv_dists[u][v]
            except KeyError or nx.NetworkXNoPath:
                continue
            given_ans = decr_constTalgo.query(u,v,true_ans)
            if not (true_ans <= given_ans and given_ans <= true_ans*(1+epsilon)):
            # if not ( given_ans <= true_ans*(1+epsilon)):
                fails += 1
                if(true_ans > given_ans):
                    print(f"FAIL: {u} {v}; T = {true_ans}, Pred:{given_ans}")
            total += 1
            
        print(f"Total: {total}, Fails: {fails}")


