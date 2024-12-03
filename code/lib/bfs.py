from lib.graphfig import * 
import networkx as nx
from collections import deque

class BFSAlgo(StatAlgo):

    def __init__(self, base_graph:nx.Graph, start_node:int):
        assert start_node in base_graph.nodes 
        super().__init__(base_graph)
        self.start_node = start_node
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
                'children' : set()
            })
        
        self.bfs_graph.nodes[start_node]['visited'] = True
        self.bfs_graph.nodes[start_node]['level'] = 0 
        self.bfs_graph.nodes[start_node]['tree_parent'] = -1 
        

        for edge in self.bfs_graph.edges:
            edge_data = self.bfs_graph.edges[edge]
            edge_data['is_tree_edge'] = False

        self.bfs_Q = deque([self.start_node]) 
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
                    
                else: 
                    self.bfs_Q.append(neighbor_node) 
                    neighbor_node_data['visited'] = True 
                    neighbor_node_data['level'] = curr_node_level + 1 
                    neighbor_node_data['parents'].add(curr_node)
                    curr_node_data['children'].add(neighbor_node)
                    neighbor_node_data['tree_parent'] = curr_node
                    self.bfs_graph.edges[(curr_node, neighbor_node)]['is_tree_edge'] = True 
    
    def path_to_source(self, node:int):
        path = [node]
        while self.bfs_graph.nodes[node]['tree_parent'] != -1:
            node = self.bfs_graph.nodes[node]['tree_parent']
            path.append(node)
        return path

class BFSVis(StatVis):

    def __init__(self, algo:BFSAlgo,fig:dict[str:go.Figure]):
        figs_dict = {"bfs_tree":fig}
        super().__init__(algo,figs_dict) 

    def node_inf_to_hovertext(self,node_data):
        return '<br>'.join([
            f"Lvl: {node_data['level']}",
            f"P_T: {node_data['tree_parent'] if node_data['tree_parent'] != -1 else 'None'}",
            f"P: {node_data['parents']   if len(node_data['parents']) > 0 else 'None'}",
            f"F: {node_data['friends']   if len(node_data['friends']) > 0 else 'None'}",
            f"C: {node_data['children']  if len(node_data['children']) > 0 else 'None'}",
        ])
    def default_init_node_visdict(self, node: int, key: str):     
        assert key == 'bfs_tree'

        nx_graph = self.graphs_dict[key] 
        node_data = nx_graph.nodes[node]

        node_data['color'] = 'green' if node == self.algo_nx.start_node else 'red'
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
        nx_positions = nx.bfs_layout(nx_graph, self.algo_nx.start_node,align='horizontal')

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

    base_graph = nx.random_geometric_graph(60,0.15)
    base_graph = base_graph.subgraph(nx.node_connected_component(base_graph,0)) 
    bfs_algo = BFSAlgo(base_graph, 0)
    fig = new_default_fig()
    fig.update_layout(hoverlabel=dict(font_size=18))
    fig.update_layout(width=1500,height=900)
    bfs_vis = BFSVis(bfs_algo,fig)
    bfs_vis.vis_init_all()

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])
    app.run_server(debug=True)