from graphfig import * 
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
    
    def get_bfs_tree(self):
        return self.bfs_graph
    
    def level(self, node:int):
        return self.bfs_graph.nodes[node]['level'] 
    
    def parent(self, node:int):
        return self.bfs_graph.nodes[node]['tree_parent'] 
    
    def is_tree_edge(self, edge:tuple):
        return self.bfs_graph.edges[edge]['is_tree_edge']
    
    def path_to_source(self, node:int):
        path = [node]
        while self.bfs_graph.nodes[node]['tree_parent'] != -1:
            node = self.bfs_graph.nodes[node]['tree_parent']
            path.append(node)
        return path

class BFSVis(StatVis):

    def __init__(self, algo:BFSAlgo,figs_dict:dict[str:go.Figure]):
        super().__init__(algo,figs_dict)

    def default_init_node_visdict(self, node: int, key: str):
        if(key == 'base'):
            super().default_init_node_visdict(node, key)
        else:
        
            nx_graph = self.graphs_dict[key] 
            node_data = nx_graph.nodes[node]
            node_data['color'] = DEFAULT_NODE_COLOR
        node_data['size'] = DEFAULT_NODE_SIZE
        node_data['text_size'] = DEFAULT_TEXT_SIZE
        node_data['hoverinfo'] = 'none'
        node_data['hovertext'] = None
        node_data['name'] = str(node)




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

    bfs_algo = BFSAlgo(base_graph, 0)
    pprint(dict(bfs_algo.bfs_graph.nodes(data=True)))

    print(bfs_algo.bfs_graph.edges(data=True))
    # pprint(dict())