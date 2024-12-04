'''
See 
Awerbuch, B., Berger, B., Cowen, L., & Peleg, D. (1998). 
Near-Linear Time Construction of Sparse Neighborhood Covers. SIAM Journal on Computing, 28(1), 263-277. 
https://doi.org/10.1137/S0097539794271898

'''


from graphfig import * 
from bfs import * 
import networkx as nx 
import random 
random.seed(31415)

'''
calX  tl_clusters
calY  ph_clusters
R     ph_init_v_notin_krs
U     ph_v_notin_cltrs 
calZ  ph_in_cltrs 
psi   kr 
psi_R R_v_in_krs  
Z     bfs_of_G_excl_calY 
psi_Z kr_of_Z
'''

def get_cluster(R:list[int],U:list[int], v:int,calY:list[list[int]],G:nx.Graph,r:float|int,beta:float):
    G2_nodes = list(set(G.nodes) - set(set.union(*calY)))
    psi_z = [v,]
    T_z = BFSAlgo(G2,)

if __name__ == "__main__":
    from pprint import pprint
    base_graph = nx.Graph() 
    base_graph.add_nodes_from(list(range(9))) 
    base_graph.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 6),
        (2, 7),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8)
    ])
    
    fig = new_default_fig()

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])
    app.run_server(debug=True)