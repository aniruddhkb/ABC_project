'''
Computing Almost Shortest Paths

Michael Elkin. 2005. 
Computing almost shortest paths. ACM Trans. Algorithms 1, 2 (October 2005), 283–323. 
https://doi.org/10.1145/1103963.1103968

Given a graph G, its subgraph H is an (alpha, beta) spanner of G iff the shortest path in H between any two nodes 

is at most alpha times the shortest path in G PLUS beta, for that same pair 

d_H(u,v) <= alpha*d_G(u,v) + beta

Typically alpha is defined as 1 + epsilon 

Given kappa and W, in Awerbuch's nbhd-cover algorithm we get a list CC of clusters, each C,  

Diameter of CC is max diameter of any graph induced by C in CC over G

1. set().union(*CC) == V
2. Diameter of CC <= uni_const * kappa * W
3. Sum of len(C.nodes) for C in CC <= O(n^(1+1/kappa))
4. Sum of len(C.edges) for C in CC <= O(mn^(1/kappa)
5. For every pair of nodes u,v in G, if d_G(u,v) <= W, then there is a C in CC 
    such that a shortest path in G between u and v is in the graph induced by C over G.
6. Every cluster C has a subset ker_C such that the ker_C_s form a partition of V. 

'''
from itertools import combinations
from math import ceil, log
from time import perf_counter

from tqdm import tqdm 
import networkx as nx 
from graphfig import * 
from bfs import * 
import fast_sparse_covers as fsc 

def get_intercl_shortest_paths(G_prime,sub_chi:list[set],distance_bound:int|float):
    cluster_bfs_graphs: list[nx.Graph] = []
    cluster_bfs_nodes: list[set] = []
    # ||
    for cluster_nodes in sub_chi: 
        cluster_bfs_graphs.append(BFSAlgo(G_prime,cluster_nodes,distance_bound).bfs_graph)
        cluster_bfs_nodes.append(set(cluster_bfs_graphs[-1].nodes))
    
    edges_to_add = []

    for i in range(len(sub_chi)):
        for j in range(i): 
            cluster_i_nodes, cluster_j_nodes = sub_chi[i], sub_chi[j] 
            if(len(cluster_i_nodes.intersection(cluster_j_nodes)) > 0):
                continue 
            bfs_i_graph = cluster_bfs_graphs[i]
            bfs_i_nodes = cluster_bfs_nodes[i] 
            inters = bfs_i_nodes.intersection(cluster_j_nodes)
            if len(inters) > 0:
                while True:
                    child_v = min([(v, bfs_i_graph.nodes[v]["level"]) for v in inters],key=lambda x: x[1])[0]
                    parent_v = bfs_i_graph.nodes[child_v]["tree_parent"]
                    edges_to_add.append((parent_v,child_v))
                    if(child_v in cluster_i_nodes):
                        break
                    child_v = parent_v
    return set(edges_to_add)


def recur_spanner(G_prime:nx.Graph,kappa:int,nu:float,D:int,K:int, recursion_depth:int = 0): 
    # print(f"Recursion depth: {recursion_depth}")
    # print(f"Number of nodes: {len(G_prime.nodes)}")
    # print()
    G_prime_nodes = set(G_prime.nodes)
    G_prime_edges = set(G_prime.edges)  
    if len(G_prime_nodes) <= K:
        part_spanner =  G_prime_edges
        return part_spanner
    else:
        assert kappa > 1 and 0 < nu and nu < (0.5 - 1/kappa) and K > 1
        part_spanner = set()
        l = ceil( log(log(len(G_prime_nodes),K),1/(1-nu)))
        
        D_pwr_l = D**l 
        
        ker_chi, chi, clusters_edges = fsc.func_get_complete_cover(G_prime,r=D_pwr_l,beta=kappa)
        assert set().union(*chi) == G_prime_nodes
        part_spanner = part_spanner.union(clusters_edges)
        len_clust_thr = len(G_prime_nodes)**(1 - nu)
        # print(f"Nbhd size: {D_pwr_l}")
        # print(f"Number of clusters: {len(chi)}")
        # print(f"Number of clusters with at least {len_clust_thr} nodes: {len([cl for cl in chi if len(cl) >= len_clust_thr])}")
        # print(f"Cluster lengths:{[len(cl) for cl in chi]}")
        chi_h = [cl for cl in chi if len(cl) >= len_clust_thr]
        chi_l = [cl for cl in chi if not cl in chi_h]
        part_spanner = part_spanner.union(get_intercl_shortest_paths(G_prime,chi_h,D*D_pwr_l))
        # input("Press Enter to continue...")
        for cl in chi_l:
            #||
            part_spanner = part_spanner.union(recur_spanner(nx.induced_subgraph(G_prime,cl),kappa,nu,D,K, recursion_depth=recursion_depth+1))
        return part_spanner

def get_parameters_for_spanner(n:int,epsilon:float, rho:float, zeta:float, c0:float = 1e-3): 

    assert all([0 < i and i < 1 for i in [epsilon,rho,zeta]]) 
    assert zeta/2 < rho and rho < zeta/2 + 1/3 

    kappa = 2/(zeta*(rho - zeta/2)) 
    nu = rho - zeta/2 
    D_coeff = ceil(8*c0/(epsilon*zeta*(rho - zeta/2))) 
    D_log_term = ceil(log(zeta/2,1-(rho - zeta/2))) 
    D = D_coeff*D_log_term
    K = ceil(n**(zeta/2)) 

    return (kappa,nu,D,K)

def validate_spanner(main_graph:nx.Graph,spanner_graph:nx.Graph,epsilon:float,beta:float):

    assert set(main_graph.nodes) == set(spanner_graph.nodes) 
    assert nx.is_connected(main_graph)
    assert nx.is_connected(spanner_graph)

    main_graph_apsp = dict(nx.shortest_path_length(main_graph))
    spanner_graph_apsp = dict(nx.shortest_path_length(spanner_graph))
    
    failures = []
    all_combinations = list(combinations(main_graph.nodes,2))
    print("\n")
    for i,j in tqdm(all_combinations):
        # assert spanner_graph_apsp[i][j] <= ceil((1 + epsilon)*main_graph_apsp[i][j] + beta)
        if not spanner_graph_apsp[i][j] <= ceil((1 + epsilon)*main_graph_apsp[i][j] + beta):
            print(f"\rFailure rate: {len(failures)/len(all_combinations)}", end="")
            # print(f"Main graph shortest path: {main_graph_apsp[i][j]}")
            # print(f"Spanner graph shortest path: {spanner_graph_apsp[i][j]}")
            # print(f"Maximum allowable shortest path: {(1 + epsilon)*main_graph_apsp[i][j] + beta}")
            failures.append((i,j))
    print("\n")
    print(f"Total failures: {len(failures)}")
    print(f"Total pairs: {len(all_combinations)}")
    print(f"Failure rate: {len(failures)/len(all_combinations)}")
    # except AssertionError:
    #     print(f"Failed for {i} and {j}")
    #     print(f"Main graph shortest path: {main_graph_apsp[i][j]}")
    #     print(f"Spanner graph shortest path: {spanner_graph_apsp[i][j]}")
    #     print(f"Maximum allowable shortest path: {(1 + epsilon)*main_graph_apsp[i][j] + beta}")
    #     raise AssertionError

def get_beta(n:int,epsilon:float,rho:float,zeta:float,c0:float = 1e-3): 
    #rho_zeta
    rosetta = ceil( log(zeta/2,1-(rho - zeta/2))) 
    base = ceil(8*c0/(epsilon*zeta*(rho - zeta/2))) 
    return (( base**(rosetta + 1) ) * ceil(log(zeta/2, (1-(rho - zeta/2))))**rosetta)

if __name__ == "__main__": 
    # from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL

    main_graph: nx.Graph = get_connected_gnp_graph(10000,5000,3e-4)
    assert len(main_graph.edges) < 20000
    print (f"Base graph has {len(main_graph.nodes)} nodes and {len(main_graph.edges)} edges")
    
    epsilon=0.6
    zeta=0.99999
    rho= zeta/2 + 0.3333
    c0=1e-6
    kappa, nu, D, K =get_parameters_for_spanner(main_graph.number_of_nodes(),epsilon=epsilon,rho=rho,zeta=zeta,c0=c0)
    beta = get_beta(main_graph.number_of_nodes(),epsilon=epsilon,rho=rho,zeta=zeta,c0=c0)
    print("rho:",rho)
    print("kappa:",kappa)
    print("nu:",nu)
    print("D:",D)
    print("K:",K)
    print("EPSILON, BETA:",epsilon,beta)
    input("Press Enter to continue...")
    

    
    
    start = perf_counter()
    spanner_edges = recur_spanner(main_graph,kappa,nu,D,K)
    end = perf_counter()
    print(f"Time taken: {end - start}")

    spanner_graph = nx.Graph(spanner_edges)
    
    print(f"Spanner graph has {len(spanner_graph.nodes)} nodes and {len(spanner_graph.edges)} edges")

    validate_spanner(main_graph,spanner_graph,epsilon,beta)
    # main_fig = default_new_fig()
    # main_sv = StatVis(StatAlgo(main_graph),{"base":main_fig})
    # main_sv.vis_init_all()
    # spanner_fig = default_new_fig()
    # spanner_sv = StatVis(StatAlgo(spanner_graph),{"base":spanner_fig})
    # spanner_sv.vis_init_all()
    # app = Dash(__name__)
    # app.layout = html.Div([
    #     dcc.Graph(id="main_graph",figure=main_fig),
    #     dcc.Graph(id="spanner_graph",figure=spanner_fig)
    # ])
    # app.run_server(debug=True,use_reloader=False)
    

    
