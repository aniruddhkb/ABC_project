'''
Conditions on the clusters (Y)_s: 

1, Every v has its r-nbhd completely contained in EXACTLY one cluster. 
2. Overlap #define as max no' of clusters that contain a single v, less than beta times the beta'th root of n 
3. For every element in some ker(~) the r-nbhd of that element is contained in ~ .  
4. All the kers are disjoint and yet cover the entire set.

Invariants through _a single phase_

1. All Ys in a phase disjoint. 
2. All kerZs in a phase disjoint.
3. Z can overlap. 
4. For every element in some ker(~) the r-nbhd of that element is contained in ~ .  
5. The union of all Ys is a subset of the union of all kerZs . 
6. The ker(Z), Z relationship and the ker(Y), Y relationship are maintained -- as well as the Y_s, kerZ_s relationship mentioned above.

'''

from itertools import combinations, product
from math import ceil
import sys
from time import perf_counter

from tqdm import tqdm 
sys.path.append('./lib') 

import networkx as nx 
from graphfig import * 
from bfs import * 
def func_get_cluster(R:set[int],U:set[int],v:int,Y_s:set[set[int]],G:nx.Graph,r:int,beta:int|float):
    
    n = len(G.nodes)

    G_notin_Y = G.copy()
    if len(Y_s) > 0:
            G_notin_Y.remove_nodes_from(list(set.union(*Y_s)))
    
    ker_z = set([v,])
    assert set(G_notin_Y.edges).issubset(set(G.edges))
    init_T_z_graph:nx.Graph = BFSAlgo.func_bfs(G_notin_Y,ker_z,r)
    assert [G_notin_Y.has_edge(*edge) for edge in init_T_z_graph.edges]
    assert [G.has_edge(*edge) for edge in init_T_z_graph.edges]
    
    z = set(init_T_z_graph.nodes)
    T_y_edges = set()
    cluster_iters = 0
    T_z_graph__double = init_T_z_graph
    while True: 
        ker_y, y, T_y = ker_z.copy(), z.copy(), T_z_graph__double
        T_y_edges = T_y_edges.union(set(T_y.edges))
        T_z_graph__double = BFSAlgo.func_bfs(G_notin_Y,y,2*r)
        T_z_graph__single = BFSAlgo.func_bfs(G_notin_Y,y,r)
        z = set(T_z_graph__double.nodes)
        ker_z = set([v for v in U.intersection(set(T_z_graph__single.nodes))]) 

        n_root_beta = n**(1/beta) 
        lenR_root_beta = len(R)**(1/beta)

        A1 = len(z)
        A2 = n_root_beta*len(y) 
        B1 = len(ker_z) 
        B2 = len(ker_y)*lenR_root_beta
        C1 = sum(dict(G.degree(z)).values()) 
        C2 = sum(dict(G.degree(y)).values())*n_root_beta 

        if(A1 <= A2 and B1 <= B2 and C1 <= C2):
            break 
        cluster_iters += 1
    return (ker_y, y, ker_z, z, T_y_edges)

def func_get_partial_cover(R:set[int], G:nx.Graph, r:int, beta:int):
    U = R.copy() 
    ker_R = set() 
    ker_Y_s, Y_s, Z_s = [], [], []
    i = 0
    T_y_s_edges = set()
    while len(U) > 0:
        i += 1
        v = next(iter(U))
        ker_y, y, ker_z, z, T_y_edges = func_get_cluster(R,U,v,Y_s,G,r,beta)
        assert all([G.has_edge(*edge) for edge in T_y_edges])
        T_y_s_edges = T_y_s_edges.union(T_y_edges)
        ker_R = ker_R.union(ker_y)
        
        U = U.difference(ker_z)
        ker_Y_s.append(ker_y)
        Y_s.append(y)
        Z_s.append(z)
        
    
    return (ker_R, ker_Y_s, Y_s, Z_s,T_y_s_edges)

def func_get_complete_cover(G,r,beta): 
    R = set(G.nodes)
    CHI = []
    KER_CHI = []
    ZEES = []
    CLUSTERS_EDGES = set()
    while len(R) > 0:

        ker_R,ker_Y_s, Y_s, Z_s, T_y_s_edges =  func_get_partial_cover(R,G,r,beta)
        CHI.extend(Y_s)
        KER_CHI.extend(ker_Y_s)
        ZEES.extend(Z_s)
        R = R.difference(ker_R)
        assert all([G.has_edge(*edge) for edge in T_y_s_edges])
        CLUSTERS_EDGES = CLUSTERS_EDGES.union(T_y_s_edges)
    
    return KER_CHI, CHI, CLUSTERS_EDGES

def get_connected_gnp_graph(n, lower_bound_n, p): 
    pre_base_graph =  nx.fast_gnp_random_graph(n,p)
    cc_nodes_lst = list(nx.connected_components(pre_base_graph))
    cc_lens_lst = [len(i) for i in cc_nodes_lst]
    idx = cc_lens_lst.index(max(cc_lens_lst)) 
    base_graph = nx.induced_subgraph(pre_base_graph,cc_nodes_lst[idx]).copy()
    try:
        assert len(base_graph.nodes) > lower_bound_n
    except AssertionError:
        
        print(max([len(i) for i in nx.connected_components(pre_base_graph)]))
        raise AssertionError

    return base_graph

def validate_clusters(G:nx.Graph, ker_Y_s:list[set[int]], Y_s:list[set[int]], r:int, beta:int): 
    V = set(G.nodes) 
    n = len(V)
    overlap_limit = beta*(n**(1/beta))
    
    print("Validating clusters. (Ys and ker_Ys relationship.)")
    
    try:
        assert set.union(*Y_s) == V 
    except AssertionError:
        print(V - set.union(*Y_s))
        raise AssertionError
    assert set.union(*ker_Y_s) == V

    assert len(Y_s) == len(ker_Y_s)
    
    print("Validating ker_Y_s disjointness.")
    for k1, k2 in tqdm(list(combinations(ker_Y_s,2))):
        assert k1.isdisjoint(k2) 
    
    print("Validating Y_s and ker_Y_s nbhd properties.")
    for ker_y, y in tqdm(list(zip(ker_Y_s,Y_s))):
        assert set(BFSAlgo.func_bfs(G,ker_y,r).nodes).issubset(y)
    
    print("Validating overlap property.")
    max_overlap = 0
    for v in tqdm(V):
        count = 0
        for y in Y_s:
            if v in y:
                count += 1 
        assert count <= overlap_limit
        if count > max_overlap:
            max_overlap = count
    print(f"Max overlap: {max_overlap}")

if __name__ == "__main__": 

    r = 4
    beta = 1
    
    # a low beta prioritizes smaller but more numerous clusters. A high beta prioritizes larger but fewer clusters.
    main_graph = get_connected_gnp_graph(1200,1000,2e-3)

    # print("Getting diameter...")
    # print(nx.diameter(G))
    n = len(main_graph.nodes)
    print(f"GRAPH SIZE:n = {n}, m = {len(main_graph.edges)}, m_max_if_complete: {n*(n-1)/2}")

    # input("Press Enter to continue...")
    t1 = perf_counter()
    KER_CHI, CHI, CLUSTERS_EDGES = func_get_complete_cover(main_graph,r,beta)
    t2 = perf_counter()
    print("Time:",t2 - t1)
    lens_chi = [len(i) for i in CHI]
    lens_ker_chi = [len(i) for i in KER_CHI]
    sum_len_chi = sum([len(cluster) for cluster in CHI])     
    print("len_chi:", len(CHI))
    print(f"sum_len_chi / n = {sum_len_chi/n}")
    print(f"lens_ker_chi:")
    print(lens_ker_chi)
    print(f"lens_chi:")
    print(lens_chi)
    validate_clusters(main_graph,KER_CHI,CHI,r,beta)

