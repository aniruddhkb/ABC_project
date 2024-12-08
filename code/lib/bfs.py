'''
Supports multiple roots, depth limit, and restricting node set.
'''

from typing import Iterable
from graphfig import * 
import networkx as nx
from collections import deque
from tqdm import tqdm

# class BFSAlgov2(StatAlgo):
#     def __init__(self, base_graph:nx.Graph, start_node_arg:int|list[int],max_level:int = None, allowed_node_set:None|set = None,copy=False):
        
#         if not max_level is None:
#             self.max_level = max_level
#         if isinstance(start_node_arg, Iterable):
#             self.multi = True
#             self.multi_roots = start_node_arg    
#         elif isinstance(start_node_arg, int):
#             self.multi = False
#             self.multi_roots = [start_node_arg,]
        
#         (self.bfs_graph, self.nx_bfs_nodes) = BFSAlgov2.func_bfs(base_graph, start_node_arg, max_level, allowed_node_set)

#         assert isinstance(self.bfs_graph, nx.DiGraph)
        
#         nx.set_edge_attributes(self.bfs_graph,True,'is_tree_edge')
        
#         for node in self.bfs_graph.nodes:
#             node_data = self.bfs_graph.nodes[node]
#             node_data.update({
#                 'visited': False, 
#                 'level': None, 
#                 'tree_parent': None,
#                 'parents' : set(),
#                 'friends' : set(),
#                 'children' : set(),
#                 'root': None,
#             })
#         for a_root in self.multi_roots:
#             self.bfs_graph.nodes[a_root].update({
#                 "visited": True,
#                 "level": 0,
#                 "tree_parent": -1,
#                 "root": a_root
#             })


#         nx.set_edge_attributes(base_graph, False, 'is_tree_edge') 

#         self.Q = deque(self.multi_roots)
#         for root in self.multi_roots:
#             node_data = self.bfs_graph.nodes[root]
#             node_data['level'] = 0
#             node_data['visited'] = True
#             node_data['root'] = root
        
#         '''
#         At the moment, only tree edges!
#         '''
#         while len(self.Q)>0:
#             node = self.Q.popleft()
#             node_data = self.bfs_graph.nodes[node] 
#             # assert node_data['visited'] 
#             for (u,v) in self.bfs_graph.out_edges(node):
#                 # assert u == node 
#                 # assert self.bfs_graph.edges[(u,v)]['is_tree_edge']
#                 v_data = self.bfs_graph.nodes[v] 
#                 v_data['level'] = node_data['level'] + 1 
#                 v_data['visited'] = True
#                 v_data['tree_parent'] = node
#                 v_data['root'] = node_data['root']
#                 v_data['parents'].add(node) 
#                 node_data['children'].add(v) 
#                 self.Q.append(v)
#         self.bfs_graph = self.bfs_graph.to_undirected()
#         self.bfs_graph.add_edges_from(list(set(base_graph.subgraph(self.nx_bfs_nodes).edges).difference(set(self.bfs_graph.edges))),is_tree_edge=False)
#         super().__init__(self.bfs_graph, copy)
#         self.all_graphs['bfs_tree'] = self.bfs_graph
        
#         for (u,v) in (self.bfs_graph.edges):
#             if not self.bfs_graph.edges[(u,v)]['is_tree_edge']: 
#                 u_data , v_data = self.bfs_graph.nodes[u], self.bfs_graph.nodes[v] 
#                 if(u_data['level'] < v_data['level']):
#                     u_data['children'].add(v)
#                     v_data['parents'].add(u) 

#                 elif(u_data['level'] > v_data['level']):
#                     u_data['parents'].add(v)
#                     v_data['children'].add(u)
                    
#                 else:
#                     u_data['friends'].add(v)
#                     v_data['friends'].add(u)

#     @staticmethod
#     def func_bfs(base_graph:nx.Graph,start_node_arg:Iterable[int]|int,max_level:int = None, allowed_node_set:None|Iterable = None)->tuple[nx.Graph,set[int]]:
        
#         allowed_node_set = set(allowed_node_set).intersection(set(base_graph.nodes)) if allowed_node_set is not None else None
#         if isinstance(start_node_arg, int):
#             if allowed_node_set is not None:
#                 bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph.subgraph(allowed_node_set), start_node_arg, depth_limit= max_level)
#             else:
#                 bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph, start_node_arg, depth_limit= max_level)

#             bfs_nodes = set(bfs_tree.nodes)
#         else:
#             virt_node = 'virtual_root' 
#             base_graph.add_node(virt_node) 
#             for node in start_node_arg:
#                 base_graph.add_edge(virt_node, node)
#             if allowed_node_set is not None:
#                 allowed_node_set.add(virt_node)
#                 bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph.subgraph(allowed_node_set), virt_node, depth_limit = max_level + 1 if max_level is not None else None)
                
#             else:
#                 bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph, virt_node, depth_limit = max_level + 1 if max_level is not None else None)
#             bfs_tree.remove_node(virt_node)
#             base_graph.remove_node(virt_node)
#             bfs_nodes = set(bfs_tree.nodes)
            
#         if allowed_node_set is not None:
#             assert bfs_nodes.issubset(allowed_node_set)
#         return bfs_tree, bfs_nodes

    
#     def path_to_source(self, start_node:int):
#         curr_node = start_node
#         path = [curr_node,]
#         while self.bfs_graph.nodes[curr_node]['tree_parent'] != -1:
#             curr_node = self.bfs_graph.nodes[curr_node]['tree_parent']
#             path.append(curr_node)
#         assert path[-1] == self.bfs_graph.nodes[start_node]['root']
#         return path

class BFSAlgo(StatAlgo):

    def __init__(self, base_graph:nx.Graph, start_node_arg:int|list[int],max_level:int|None = None, allowed_node_set:None|set = None):
        super().__init__(base_graph)
        self.max_level = max_level
        self.allowed_node_set = allowed_node_set

        if isinstance(start_node_arg, Iterable):
            self.multi = True
            self.multi_roots = start_node_arg
            assert all([node in base_graph.nodes for node in self.multi_roots])
        elif isinstance(start_node_arg, int):
            self.multi = False
            assert start_node_arg in base_graph.nodes
            self.multi_roots = (start_node_arg,)
        
        
        if self.allowed_node_set is not None:
            assert all([node in self.allowed_node_set for node in self.multi_roots])
        
        self.bfs_graph = self.base_graph.copy()
        self.all_graphs['bfs_tree'] = self.bfs_graph
        for node in self.bfs_graph.nodes:
            node_data = self.bfs_graph.nodes[node]
            node_data.update({
                'visited': False, 
                'level': None, 
                'tree_parent': None,
                'parents' : set(),
                'friends' : set(),
                'children' : set(),
                'root': None,
            })
        for a_root in self.multi_roots:
            self.bfs_graph.nodes[a_root]['visited'] = True
            self.bfs_graph.nodes[a_root]['level'] = 0 
            self.bfs_graph.nodes[a_root]['tree_parent'] = -1
            self.bfs_graph.nodes[a_root]['root'] = a_root 
        
        self.max_level = max_level
        #print(self.max_level is None)
        for edge in self.bfs_graph.edges:
            edge_data = self.bfs_graph.edges[edge]
            edge_data['is_tree_edge'] = False

        self.bfs_safe_nodes = set(self.multi_roots)
        self.bfs_Q = deque(self.multi_roots) 
        #print("maxlevel", self.max_level)

        while len(self.bfs_Q) > 0:

            curr_node = self.bfs_Q.popleft() 
            curr_node_data = self.bfs_graph.nodes[curr_node]
            curr_node_level = curr_node_data['level']
            
            for neighbor_node in self.bfs_graph.neighbors(curr_node):
                neighbor_node_data = self.bfs_graph.nodes[neighbor_node] 

                if neighbor_node_data['visited']:
                    neighbor_node_level = neighbor_node_data['level']
                    
                    if neighbor_node_level == curr_node_level + 1: 
                        neighbor_node_data['parents'].add(curr_node)
                        curr_node_data['children'].add(neighbor_node)

                    elif neighbor_node_level == curr_node_level:
                        curr_node_data['friends'].add(neighbor_node)
                        neighbor_node_data['friends'].add(curr_node)

                    elif neighbor_node_level == curr_node_level - 1: 
                        curr_node_data['parents'].add(neighbor_node)
                        neighbor_node_data['children'].add(curr_node)
                    else:
                        raise ValueError(f'Inconsistent levels for nodes {curr_node} and {neighbor_node}')
                    
                elif all([(  (self.max_level is not None) and (curr_node_level + 1 <= self.max_level)) or self.max_level is None,
                          (self.allowed_node_set is None) or (neighbor_node in self.allowed_node_set)
                          ]): 
                    #print("YEE")
                    self.bfs_Q.append(neighbor_node)

                    curr_node_data['children'].add(neighbor_node)
                    self.bfs_graph.edges[(curr_node, neighbor_node)]['is_tree_edge'] = True 
                    
                    neighbor_node_data['visited'] = True 
                    neighbor_node_data['level'] = curr_node_level + 1 
                    neighbor_node_data['tree_parent'] = curr_node
                    neighbor_node_data['parents'].add(curr_node)
                    neighbor_node_data['root'] = curr_node_data['root']
                    self.bfs_safe_nodes.add(neighbor_node)
                # else:
                    #print(neighbor_node, neighbor_node in self.allowed_node_set, ((self.max_level is not None) and (curr_node_level + 1 <= self.max_level)) or self.max_level is None )
        self.bfs_graph = self.bfs_graph.subgraph(self.bfs_safe_nodes).copy()

    def func_bfs(base_graph:nx.Graph,start_node_arg:Iterable[int]|int,max_level:int = None, allowed_node_set:None|Iterable = None)->nx.DiGraph:
        
        allowed_node_set = set(allowed_node_set).intersection(set(base_graph.nodes)) if allowed_node_set is not None else None
        if isinstance(start_node_arg, int):
            if allowed_node_set is not None:
                bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph.subgraph(allowed_node_set), start_node_arg, depth_limit= max_level)
            else:
                bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph, start_node_arg, depth_limit= max_level)

            bfs_nodes = set(bfs_tree.nodes)
        else:
            virt_node = 'virtual_root' 
            base_graph.add_node(virt_node) 
            for node in start_node_arg:
                base_graph.add_edge(virt_node, node)
            if allowed_node_set is not None:
                allowed_node_set.add(virt_node)
                bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph.subgraph(allowed_node_set), virt_node, depth_limit = max_level + 1 if max_level is not None else None)
                
            else:
                bfs_tree:nx.DiGraph = nx.bfs_tree(base_graph, virt_node, depth_limit = max_level + 1 if max_level is not None else None)
            bfs_tree.remove_node(virt_node)
            base_graph.remove_node(virt_node)
            bfs_nodes = set(bfs_tree.nodes)
            
        if allowed_node_set is not None:
            assert bfs_nodes.issubset(allowed_node_set)
        return bfs_tree
    
    
    def path_to_source(self, start_node:int):
        curr_node = start_node
        path = [curr_node,]
        while self.bfs_graph.nodes[curr_node]['tree_parent'] != -1:
            curr_node = self.bfs_graph.nodes[curr_node]['tree_parent']
            path.append(curr_node)
        assert path[-1] == self.bfs_graph.nodes[start_node]['root']
        return path

class BFSVis(StatVis):

    # def __init__(self, algo:BFSAlgo|BFSAlgov2,fig:go.Figure):
    def __init__(self, algo:BFSAlgo,fig:go.Figure):
        figs_dict = {"bfs_tree":fig}
        super().__init__(algo,figs_dict) 
        
        # assert isinstance(algo, BFSAlgo) or isinstance(algo, BFSAlgov2)
        assert isinstance(algo, BFSAlgo)
        self.bfs_algo = algo
        self.multi = self.bfs_algo.multi

    def node_inf_to_hovertext(self,node_data):
        return '<br>'.join([
            f"Lvl: {node_data['level']}",
            f"P_T: {node_data['tree_parent'] if node_data['tree_parent'] != -1 else 'None'}",
            f"P: {node_data['parents']   if len(node_data['parents']) > 0 else 'None'}",
            f"F: {node_data['friends']   if len(node_data['friends']) > 0 else 'None'}",
            f"C: {node_data['children']  if len(node_data['children']) > 0 else 'None'}",
            f"Root: {node_data['root']}" if self.multi else ''
        ])
    
    def default_init_node_visdict(self, node: int, key: str):     
        assert key == 'bfs_tree'

        nx_graph = self.graphs_dict[key] 
        node_data = nx_graph.nodes[node]

        node_data['color'] = 'green' if node in self.bfs_algo.multi_roots else 'red'
        node_data['size'] = DEFAULT_NODE_SIZE
        node_data['text_size'] = DEFAULT_TEXT_SIZE
        node_data['hoverinfo'] = None 
        node_data['hovertext'] = self.node_inf_to_hovertext(node_data)      
        node_data['name'] = str(node)

    def default_init_edge_visdict(self, u:int, v:int , key: str):
        
        assert key == 'bfs_tree'
        nx_graph = self.graphs_dict[key]
        u,v = min(u,v), max(u,v)
        edge = (u,v)
        edge_data = nx_graph.edges[edge]

        edge_data['name'] = f"{u}_{v}"
        edge_data['color'] = 'red' if edge_data['is_tree_edge'] else DEFAULT_EDGE_COLOR
        edge_data['width'] = DEFAULT_EDGE_WIDTH
        edge_data['hoverinfo'] = 'none'
        edge_data['hovertext'] = None

    def default_init_nx_layout(self, key: str) -> None:
        
        assert key == 'bfs_tree'
        nx_graph = self.graphs_dict[key]
        if(self.multi):
            vis_start_node = 'virtual_root'
            nx_graph.add_node(vis_start_node)
            for root in self.bfs_algo.multi_roots:
                nx_graph.add_edge(vis_start_node, root)
        else:
            vis_start_node = self.bfs_algo.multi_roots[0]

        nx_positions = nx.bfs_layout(nx_graph, vis_start_node,align='horizontal')
        if(self.multi):
            nx_graph.remove_node(vis_start_node)

        for node in nx_graph.nodes:
            nx_positions[node][1] = -nx_positions[node][1]
            nx_graph.nodes[node]['pos'] = nx_positions[node].tolist()

if __name__ == "__main__": 
    # from p#print import p#print
    main_graph = get_connected_gnp_graph(150,100,0.05)
    # base_graph = base_graph.subgraph(nx.node_connected_component(base_graph,0)) 
    bfs_algo = BFSAlgo(main_graph, 0)
    spanner_fig = default_new_fig()
    spanner_fig.update_layout(hoverlabel=dict(font_size=18))
    spanner_fig.update_layout(width=1500,height=900)
    es_vis = BFSVis(bfs_algo,spanner_fig)
    es_vis.vis_init_all()

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(figure=spanner_fig)
    ])
    app.run_server(debug=True)