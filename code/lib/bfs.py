'''
Supports multiple roots, depth limit, and restricting node set.
'''

from typing import Iterable
from graphfig import * 
import networkx as nx
from collections import deque

class BFSAlgo(StatAlgo):

    def __init__(self, base_graph:nx.Graph, start_node_arg:int|list[int],max_level:int = None, allowed_node_set:None|set = None):
        super().__init__(base_graph)
        if isinstance(start_node_arg, Iterable):
            self.multi = True
            self.multi_roots = start_node_arg
            assert all([node in base_graph.nodes for node in self.multi_roots])
        elif isinstance(start_node_arg, int):
            self.multi = False
            assert start_node_arg in base_graph.nodes
            self.multi_roots = (start_node_arg,)
        self.allowed_node_set = allowed_node_set
        
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

        for edge in self.bfs_graph.edges:
            edge_data = self.bfs_graph.edges[edge]
            edge_data['is_tree_edge'] = False

        self.bfs_safe_nodes = set(self.multi_roots)
        self.bfs_Q = deque(self.multi_roots) 
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
                    
                elif all([(curr_node_level + 1 <= self.max_level and self.max_level is not None) or self.max_level is None,
                          self.allowed_node_set is None or neighbor_node in self.allowed_node_set
                          ]): 
                    self.bfs_Q.append(neighbor_node) 
                    curr_node_data['children'].add(neighbor_node)
                    self.bfs_graph.edges[(curr_node, neighbor_node)]['is_tree_edge'] = True 
                    
                    neighbor_node_data['visited'] = True 
                    neighbor_node_data['level'] = curr_node_level + 1 
                    neighbor_node_data['tree_parent'] = curr_node
                    neighbor_node_data['parents'].add(curr_node)
                    neighbor_node_data['root'] = curr_node_data['root']
                    self.bfs_safe_nodes.add(neighbor_node)
        
        nodescpy = list(self.bfs_graph.nodes)
        for node in nodescpy:
            if node not in self.bfs_safe_nodes:
                self.bfs_graph.remove_node(node)
    
    def path_to_source(self, start_node:int):
        curr_node = start_node
        path = [curr_node,]
        while self.bfs_graph.nodes[curr_node]['tree_parent'] != -1:
            curr_node = self.bfs_graph.nodes[curr_node]['tree_parent']
            path.append(curr_node)
        assert path[-1] == self.bfs_graph.nodes[start_node]['root']
        return path

class BFSVis(StatVis):

    def __init__(self, algo:BFSAlgo,fig:go.Figure):
        figs_dict = {"bfs_tree":fig}
        super().__init__(algo,figs_dict) 
        
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

    base_graph = nx.random_geometric_graph(30,0.3)
    # base_graph = base_graph.subgraph(nx.node_connected_component(base_graph,0)) 
    bfs_algo = BFSAlgo(base_graph, [0,1],3)
    fig = new_default_fig()
    fig.update_layout(hoverlabel=dict(font_size=18))
    fig.update_layout(width=1500,height=900)
    es_vis = BFSVis(bfs_algo,fig)
    es_vis.vis_init_all()

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])
    app.run_server(debug=True)