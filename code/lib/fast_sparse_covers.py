'''
See 
Awerbuch, B., Berger, B., Cowen, L., & Peleg, D. (1998). 
Near-Linear Time Construction of Sparse Neighborhood Covers. SIAM Journal on Computing, 28(1), 263-277. 
https://doi.org/10.1137/S0097539794271898

'''

'''
2024_12_05 18:06 -- something's wrong here. Persistent assertion failure -- trying to launch a BFS from node not in graph G2. 

'''


from math import log
import time
from graphfig import * 
from bfs import * 
import networkx as nx 
import random 
from tqdm import tqdm
from pprint import pprint
random.seed(31415)

'''

'''


class FastSparseCovers(StatAlgo):

    def __init__(self, base_graph:nx.Graph, r:int, beta:float):
        super().__init__(base_graph) 
        self.r = r
        self.beta = beta

    def validate_cluster(self,ker_y, y): 

        bfs_trees_to_validate = {}
        for vert in ker_y: 
            bfs_trees_to_validate[vert] = BFSAlgo(self.base_graph, vert,self.r).bfs_graph

        # Condition 1 
        for vert in ker_y: 
            for node in bfs_trees_to_validate[vert].nodes: 
                assert node in y

    def get_cluster_nodes(self,
            R:set[int],
            U:set[int], 
            seed_vert:int,
            Y:set[set[int]]):

        
        ker_z = set([seed_vert,]) 

        G2_nodes = list(set(self.base_graph.nodes).difference(set().union(*Y)))
        G2 = self.base_graph.subgraph(G2_nodes).copy()
        T_z = BFSAlgo(G2,seed_vert,self.r).bfs_graph.copy() 
        z = set(T_z.nodes)
        n_iters = 0

        while (True): 
            y, ker_y = z.copy(), ker_z.copy()
            T_z = BFSAlgo(G2,y,2*self.r).bfs_graph.copy()
            z = z.union(set(T_z.nodes))
            
            ker_z = set() 

            inters = z.intersection(U)
            for v in inters: 
                if(T_z.nodes[v]['level'] <= self.r):
                    ker_z.add(v)

            A = len(z) <=  len(y)*len(self.base_graph.nodes)**(1/self.beta)
            B = len(ker_z) <= len(ker_y)*len(R)**(1/self.beta) 
            C = sum([i[1] for i in self.base_graph.degree(z)]) <= sum([i[1] for i in self.base_graph.degree(y)])*len(self.base_graph.nodes)**(1/self.beta)

            if all([A, B, C]):
                break
            n_iters += 1
        return ker_y, y, ker_z, z
            
    def get_cover_once_nodes(self,R:set[int]): 
        U = R.copy()
        kerlike_R = set() 
        Y, Z, kers_Y, kers_Z = [list(),]*4 
        i = 0
        while len(U) > 0:
            seed_vert = next(iter(U))
            ker_y, y, ker_z, z = self.get_cluster_nodes(R,U,seed_vert,Y) 
            kerlike_R = kerlike_R.union(ker_y) 
            Y.append(y) 
            Z.append(z) 
            U = U.difference(ker_z)
            kers_Y.append(ker_y)
            kers_Z.append(ker_z)
            i += 1
        return kerlike_R, Y, kers_Y, Z, kers_Z 
    
    def make_cover_all_nodes(self): 
        R = set(self.base_graph.nodes)
        self.clusters_lst = []
        while len(R) > 0:
            kerlike_R, Y, kers_Y, Z, kers_Z = self.get_cover_once_nodes(R)
            self.clusters_lst.extend(Y) 

            R = R.difference(kerlike_R)

if __name__ == "__main__":
    
    base_graph = nx.fast_gnp_random_graph(200,5)

    fsc = FastSparseCovers(base_graph,r=2,beta=5)
    
    fsc.make_cover_all_nodes()
    for cluster in fsc.clusters_lst: 
        pprint(cluster)
    # time.sleep(1)
    # fig1 = new_default_fig()
    
    # vis = StatVis(StatAlgo(fsc.base_graph),{"base":fig1})
    # vis.vis_init_all()
    

    # app = Dash(__name__)

    # app.layout = html.Div([
    #     dcc.Graph(figure=fig1),

    # ])

    # app.run_server(debug=True)