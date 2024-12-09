from graphfig import *
from bfs import *
from even_shiloach import *
from recursive_spanner import *
from decr_const import *
import networkx as nx
'''
Parameters for each:

TUNED 

c <-- proportionality constant

epsilon <-- error parameter

rho zeta c0 <-- parameters for spanner

d <-- distance parameter NONCONSTANT
t <-- number of insertion centers in current phase. NONCONSTANT

spanner_roots = subset of size cn ln n / (epsilon d) 
Make multibfs of spanner from these.  

decr_const using c and epsilon.

insertion_centers = []
d levels of single-ES from each insertion center. (d nonconstant)

Update(delete):
    Remake spanner
    Delete-op decr_const
    Delete-op each insertion center 

Update(insert):
    if len(insertion_centers) >= t:
        Refresh ALL 
    
    Add insertion center 
    Make its ES tree
    
Query u v
    l1 <- decr_const.query 
    l3 <- spanner.query 
    l2 <- 

    
for each insertion center w 

Check if u AND v both in ES tree of w.

If yes, return min over all w of delta uw + delat wv as l2

return min(l1,l2,l3)

DERIVED 

n

'''

class DynAPSPAlgo(DynAlgo):
    def __init__(self, base_graph:nx.Graph, all_epsilon:float,all_delta:float, decr_c:float, sp_rho:float, sp_zeta:float, sp_c:float): 
        super().__init__(base_graph)
        self.common_graph = self.base_graph
        self.n = self.common_graph.number_of_nodes()
        self.all_epsilon = all_epsilon
        self.all_delta = all_delta
        self.decr_c = decr_c
        
        self.sp_rho = sp_rho
        self.sp_zeta = sp_zeta
        self.sp_c = sp_c

        self.sp_kappa, self.sp_nu, self.sp_D, self.sp_K = get_parameters_for_spanner(self.n,self.all_epsilon,
                                                                                     self.sp_rho,self.sp_zeta,
                                                                                     self.sp_c)
        self.new_phase()

        self.l_mins_counts = [0,0,0]

    def get_new_spanner_tree(self):

        self.sp_n_sources = ceil(self.sp_c*self.n*log(self.n)/(self.all_epsilon*self.nonconst_d))
        if self.sp_n_sources > self.n:
            raise ValueError("Too many spanner sources.")

        self.sp_graph = nx.Graph()
        self.sp_graph.add_nodes_from(self.common_graph.nodes)
        self.sp_graph.add_edges_from(recur_spanner(self.common_graph,self.sp_kappa,self.sp_nu,self.sp_D,self.sp_K))
        assert nx.is_connected(self.sp_graph)
        assert self.common_graph.number_of_nodes() == self.n
        assert self.sp_graph.number_of_nodes() == self.n
        self.sp_roots = random.sample(list(self.sp_graph.nodes),self.sp_n_sources)
        self.sp_tree = ESAlgov2(self.sp_graph,self.sp_roots)
    
    def reinit_consts(self):
        self.m = self.common_graph.number_of_edges()
        
        self.nonconst_d = self.n**(1 + self.all_delta) / self.m**0.5 
        self.nonconst_t = floor(self.m**0.5 /self.n**self.all_delta)

    def new_phase(self):
        
        self.decr_apsp_algo = DecrAPSPConstTAlgo(self.base_graph,self.all_epsilon,self.decr_c) 

        self.reinit_consts()
        self.get_new_spanner_tree()

        self.ins_trees:dict[int:ESAlgov2] = {}
        

    def delete(self,edges_to_delete:list[tuple[int,int]]):
        
        assert all([self.common_graph.has_edge(u,v) for (u,v) in edges_to_delete])
        
        for (u,v) in edges_to_delete:
            self.decr_apsp_algo.delete(u,v)


            for w in self.ins_trees:
                ins_tree:ESAlgov2 = self.ins_trees[w]
                if(ins_tree.es_graph.has_edge(u,v)):
                    ins_tree.es_delete_oneshot(u,v)
        
            self.common_graph.remove_edge(u,v)
        self.reinit_consts()
        self.get_new_spanner_tree()

    def insert(self,node:int,new_neighbors:list[int]): 
        
        assert self.common_graph.has_node(node)
        for new_neighbor in new_neighbors:
            assert self.common_graph.has_node(new_neighbor)
            assert not self.common_graph.has_edge(node,new_neighbor)
            self.common_graph.add_edge(node,new_neighbor)
        

        self.reinit_consts()
        if(len(self.ins_trees) >= self.nonconst_t):
            self.new_phase()
        
        else:
            new_tree = ESAlgov2(self.common_graph,node,self.nonconst_d)
            self.ins_trees[node] = new_tree 
            
    def query(self,u:int,v:int): 
        
        assert u in self.common_graph.nodes and v in self.common_graph.nodes 
        if u == v:
            return 0 
        
        l1 = self.decr_apsp_algo.query(u,v)
        l2 = float("inf")
        l3 = self.sp_tree.get_level(u) + self.sp_tree.get_level(v) 

        for w in self.ins_trees:
            w_tree:ESAlgov2 = self.ins_trees[w]

            if w_tree.es_graph.has_node(u) and w_tree.es_graph.has_node(v):
                l2 = min(l2,w_tree.get_level(u) + w_tree.get_level(v))
        l_s = [l1,l2,l3]
        l_min_idx = l_s.index(min(l1,l2,l3))
        self.l_mins_counts[l_min_idx] += 1
        return l_s[l_min_idx]

if __name__ == "__main__":
    try:
        import syscheck
        syscheck.syscheck()
        MULTI_THREAD = True
    except RuntimeError:
        MULTI_THREAD = False
    N_THREADS = 11 
    base_graph = get_connected_gnp_graph(1000,500,4e-3)
    input(f"Base graph has {base_graph.number_of_nodes()} nodes and {base_graph.number_of_edges()} edges. Press Enter to continue.")
    all_epsilon = 0.6 
    all_delta = 0.5 

    decr_c = 0.5

    sp_zeta = 0.5 
    sp_rho = sp_zeta/2 + 0.3333
    sp_c = 1e-6 

    dyn_apsp_algo = DynAPSPAlgo(base_graph,all_epsilon,all_delta,decr_c,sp_rho,sp_zeta,sp_c)

    print("FIRST TESTING:")
    total = 0
    fails = 0
    true_uv_dists = dict(nx.all_pairs_shortest_path_length(dyn_apsp_algo.common_graph))
    for u,v in tqdm(list(combinations(dyn_apsp_algo.base_graph.nodes,2))):
        try: 
            true_ans = true_uv_dists[u][v]
        except KeyError or nx.NetworkXNoPath:
            continue
        given_ans = dyn_apsp_algo.query(u,v)
        if not (true_ans <= given_ans and given_ans <= true_ans*(1+all_epsilon)):
        # if not ( given_ans <= true_ans*(1+epsilon)):
            fails += 1
            
            print(f"FAIL: {u} {v}; T = {true_ans}, Pred:{given_ans}")
        total += 1
    print(f"Total: {total}, Fails: {fails}")
    
    print("DELETIONS TESTING:")
    while True:
        edges_to_delete = random.sample(list(dyn_apsp_algo.common_graph.edges),100)
        for edge in edges_to_delete:
            assert dyn_apsp_algo.common_graph.has_edge(*edge)
            dyn_apsp_algo.common_graph.remove_edge(*edge)
        for edge in edges_to_delete:
            assert not dyn_apsp_algo.common_graph.has_edge(*edge)
            dyn_apsp_algo.common_graph.add_edge(*edge)
        
        edges_to_delete_cpy = edges_to_delete.copy()
        
        for edge in edges_to_delete_cpy:
            dyn_apsp_algo.common_graph.remove_edge(*edge)
            if(not nx.is_connected(dyn_apsp_algo.common_graph)):
                dyn_apsp_algo.common_graph.add_edge(*edge)
                edges_to_delete.remove(edge)
        for edge in edges_to_delete_cpy:
            if(not dyn_apsp_algo.common_graph.has_edge(*edge)):
                dyn_apsp_algo.common_graph.add_edge(*edge)


        print("Pre-validation of deletion done.")
        dyn_apsp_algo.delete(edges_to_delete)
        true_uv_dists = dict(nx.all_pairs_shortest_path_length(dyn_apsp_algo.base_graph))
        total = 0
        fails = 0
        for u,v in tqdm(list(combinations(dyn_apsp_algo.base_graph.nodes,2))):
            try: 
                true_ans = true_uv_dists[u][v]
            except KeyError or nx.NetworkXNoPath:
                continue
            given_ans = dyn_apsp_algo.query(u,v)
            if not (true_ans <= given_ans and given_ans <= true_ans*(1+all_epsilon)):
            # if not ( given_ans <= true_ans*(1+epsilon)):
                fails += 1
                
                print(f"FAIL: {u} {v}; T = {true_ans}, Pred:{given_ans}")
            total += 1
        print(f"Total: {total}, Fails: {fails}; {len(edges_to_delete)} deletions. {dyn_apsp_algo.common_graph.number_of_edges()} edges remaining.")   


            