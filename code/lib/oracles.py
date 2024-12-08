from itertools import combinations, product
from graphfig import * 
from bfs import *
from even_shiloach import *
import networkx as nx
from collections import deque
import random 
random.seed(31415)

get_cointoss = lambda p: random.binomialvariate(1,p)

class Oracles(DynAlgo):

    def __init__(self, base_graph:nx.Graph, k:int, d:int): 
        DynAlgo.__init__(self, base_graph) 
        self.oracle_graph = self.base_graph
        self.n = len(self.oracle_graph.nodes)
        self.all_graphs["oracle_base"] = self.oracle_graph

        self.k = k
        self.d = d
        self.dbar = (2*k - 1)*d 

        self.A_s = dict()
        self.A_s[0] = set(self.oracle_graph.nodes)
        for i in range(1, self.k + 1):
            self.A_s[i] = set()

        self.B = dict()
        for i in self.oracle_graph.nodes:
            self.B[i] = set()

     
        self.C = dict()
        for i in self.oracle_graph.nodes:
            self.C[i] = set()
        
        self.cointoss_prob = self.n**(-1/self.k)
        for a_idx in range(1,self.k):
            for node in self.A_s[a_idx-1]:
                if get_cointoss(self.cointoss_prob):
                    self.A_s[a_idx].add(node)
        self.A_bar_s = dict()
        self.multi_ES_A_s = dict()
        
        for a_idx in range(self.k):
            self.A_bar_s[a_idx]= (self.A_s[a_idx] - self.A_s[a_idx+1])
            self.multi_ES_A_s[a_idx] = (ESAlgov2(self.oracle_graph,self.A_s[a_idx],self.dbar))
        
        
        self.single_ES_s:dict[int,ESAlgov2] = dict()
        for v in self.base_graph.nodes:
            self.single_ES_s[v] = ESAlgov2(self.oracle_graph,v,self.dbar)
        

        for (a_idx, v) in  list(product(range(self.k - 1),self.oracle_graph.nodes)): 
            
            for w in self.A_bar_s[a_idx]:
                T_w: ESAlgov2 = self.single_ES_s[w]
                M_nxt:ESAlgov2 = self.multi_ES_A_s[a_idx+1]

                if v in T_w.es_graph.nodes:
                    if v in M_nxt.es_graph.nodes:
                        delt_v_w = T_w.get_level(v)
                        delt_v_A_nxt = M_nxt.get_level(v)
                        if delt_v_w <= delt_v_A_nxt:
                            self.C[w].add(v)
                            self.B[v].add(w)
                    else:
                        self.C[w].add(v)
                        self.B[v].add(w)
    def double_distance_query(self,u:int,v:int)->int|float:
        return min(
            self.distance_query(u,v),
            self.distance_query(v,u)
        )
    def distance_query(self,u:int,v:int)->int|float:
        assert u in self.oracle_graph.nodes and v in self.oracle_graph.nodes
        w = u 
        a_idx = 0
        
        while not w in self.B[v]:
            a_idx += 1
            if(a_idx == self.k):
                return float("inf")
            u, v = v, u
            m_ES:ESAlgov2 = self.multi_ES_A_s[a_idx]
            w = m_ES.get_root(u)
        m_ES:ESAlgov2 = self.multi_ES_A_s[a_idx] 
        s_ES:ESAlgov2 = self.single_ES_s[v] 
        return m_ES.get_level(u) + s_ES.get_level(w)
    
    def delete(self,u:int,v:int)->None: 
        assert u in self.oracle_graph.nodes and v in self.oracle_graph.nodes

        shifted = list()

        for w in self.oracle_graph.nodes:
            s_ES_w:ESAlgov2 =  self.single_ES_s[w]
            if((u,v) in s_ES_w.es_graph.edges):
                s_ES_w.es_delete_genf(u,v,perf_mode=True)
                shifted.append((s_ES_w, s_ES_w.shifted, True)) 
        for a_idx in range(self.k):
            m_ES_a:ESAlgov2 = self.multi_ES_A_s[a_idx]
            if((u,v) in m_ES_a.es_graph.edges):
                m_ES_a.es_delete_genf(u,v,perf_mode=True)
                shifted.append((m_ES_a,m_ES_a.shifted, False))
        for shifted_triple in shifted:
            es:ESAlgov2 = shifted_triple[0]
            shifted_nodes:list[int] = shifted_triple[1]
            is_single:bool = shifted_triple[2]

            for x in shifted_nodes:
                if(is_single):
                    w = es.get_root(x)
                    abar_idx = self.which_Abar_s[w]

                    m_ES:ESAlgov2 = self.multi_ES_A_s[abar_idx+1]
                    delta_x_w = es.get_level(x)

                    if x not in m_ES.es_graph.nodes or delta_x_w < m_ES.get_level(x):
                        self.C[w].add(x)
                        self.B[x].add(w)
                    else:
                        self.C[w].remove(x)
                        self.B[x].remove(w)

                else:
                    alpha = self.multi_ES_A_s.index(es) - 1
                    for w in self.A_bar_s[alpha]:
                        delta_x_A_nxt = es.get_level(x)
                        T_w:ESAlgov2 = self.single_ES_s[w]
                        
                        if x not in T_w.es_graph.nodes or T_w.get_level(x) >= delta_x_A_nxt:
                            self.C[w].remove(x)
                            self.B[x].remove(w)
                        else:
                            self.C[w].add(x)
                            self.B[x].add(w)
                    

if __name__ == "__main__":
    from pprint import pprint
    k= 2 
    d = 3
    eps = 2*k - 1 

    nx_graph = get_connected_gnp_graph(400,200,0.008)
    oracles = Oracles(nx_graph,k,d) 
    # print(nx.diameter(nx_graph))

    # for i in oracles.oracle_graph.nodes:
    #     tree_i = oracles.single_ES_s[i].es_graph
    #     print(f"s_tree_i {i}")
    #     for node in tree_i.nodes:
    #         print(f"\tNode: {node}, Level: {tree_i.nodes[node]['level']}")
    
    # fig = default_new_fig()
    # sv = StatVis(oracles,{"base":fig})
    # sv.vis_init_all()
    # fig.show()
    

    # fig2 = default_new_fig()
    # bfs_vis = ESVisv2(oracles.multi_ES_A_s[1],fig2)
    # bfs_vis.vis_init_all()
    # fig2.show()
    total = 0
    fails = 0
    for u, v in tqdm(list(combinations(oracles.oracle_graph.nodes,2))):
        true_ans = nx.shortest_path_length(oracles.oracle_graph,u,v)
        if true_ans <= d:
            total += 1
            given_ans = oracles.distance_query(u,v)
            # print(f"\rCOMBO: {u} {v}; T = {true_ans}, Pred:{given_ans}",end="")
            if(not given_ans <= eps*true_ans ):
                fails += 1
                # print(f"FAIL: {u} {v}; T = {true_ans}, Pred:{given_ans}")
    print(f"\nTotal: {total}, Fails: {fails}")    
    print("\n")
    
                            
                    
            
        
        
        

                



