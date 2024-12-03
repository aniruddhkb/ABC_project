from lib.graphfig import * 
from lib.bfs import *
import networkx as nx
from collections import deque
import random 
random.seed(31415)


class ESAlgo(DynAlgo):

    def __init__(self, base_graph:nx.Graph, start_node:int):
        assert start_node in base_graph.nodes 
        super().__init__(base_graph)
        self.es_graph = self.base_graph
        self.start_node = start_node
        self.all_graphs['es_tree'] = self.es_graph
        for node in self.es_graph.nodes:
            node_data = self.es_graph.nodes[node]

            node_data.update({
                'visited': False, 
                'level': None, 
                'tree_parent': None,
                'parents' : set(),
                'friends' : set(),
                'children' : set()
            })
        
        self.es_graph.nodes[start_node]['visited'] = True
        self.es_graph.nodes[start_node]['level'] = 0 
        self.es_graph.nodes[start_node]['tree_parent'] = -1 
        

        for edge in self.es_graph.edges:
            edge_data = self.es_graph.edges[edge]
            edge_data['is_tree_edge'] = False

        self.bfs_Q = deque([self.start_node]) 
        while len(self.bfs_Q) > 0:
            curr_node = self.bfs_Q.popleft() 
            curr_node_data = self.es_graph.nodes[curr_node]
            curr_node_level = curr_node_data['level']
            for neighbor_node in self.es_graph.neighbors(curr_node):
                neighbor_node_data = self.es_graph.nodes[neighbor_node] 

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
                    self.es_graph.edges[(curr_node, neighbor_node)]['is_tree_edge'] = True 
    
    def path_to_source(self, node:int):
        path = [node]
        while self.es_graph.nodes[node]['tree_parent'] != -1:
            node = self.es_graph.nodes[node]['tree_parent']
            path.append(node)
        return path

    def es_delete_subgraph(self, node:int, curr_updates:dict|None):
        
        
        nodes_del_Q = deque([node]) 
        while len(nodes_del_Q) > 0:
            curr_node = nodes_del_Q.pop()
            
            if not curr_updates is None:
                curr_updates['es_tree']['nodes'].append((curr_node,'DEL'))
            
            for neighbor in self.es_graph.neighbors(curr_node):
            
                if not curr_updates is None:
                    curr_updates['es_tree']['edges'].append((curr_node,neighbor,'DEL')) 
            
                nodes_del_Q.appendleft(neighbor)
            self.es_graph.remove_node(curr_node)

    def es_update_delete_edge(self, u:int, v:int, perf_mode:bool=False):
        '''
        DOES NOT PERFORM THE DELETE OP IN THE BASE GRAPH.
        '''

        assert self.es_graph.has_edge(u,v)
        self.es_graph.remove_edge(u,v)
        u_data = self.es_graph.nodes[u] 
        v_data = self.es_graph.nodes[v] 

        if not perf_mode:
            self.update_dict = self.get_new_update_dict()
            curr_updates = self.get_new_update_dict()
            curr_updates['es_tree']['edges'].append((u,v,'DEL'))
            curr_updates['es_tree']['nodes'].append((u,'MOD'))
            curr_updates['es_tree']['nodes'].append((v,'MOD'))

        if u_data['level'] > v_data['level']: 
            u,v = v,u
            u_data,v_data = v_data,u_data 
        
        if u_data['level'] == v_data['level']:
            assert u in v_data['friends'] and v in u_data['friends']
            u_data['friends'].remove(v)
            v_data['friends'].remove(u)

        elif u_data['level'] == v_data['level'] - 1 and v_data['tree_parent'] != u:
            assert u in v_data['parents'] and v in u_data['children']
            u_data['children'].remove(v)
            v_data['parents'].remove(u)

        elif u_data['level'] == v_data['level'] - 1 and v_data['tree_parent'] == u:
            assert u in v_data['parents'] and v in u_data['children']
            u_data['children'].remove(v)
            v_data['parents'].remove(u)

            if len(v_data['parents']) > 0:
                v_data['tree_parent'] = v_data['parents'].pop() 
                v_data['parents'].add(v_data['tree_parent'])
                self.es_graph.edges[(v_data['tree_parent'],v)]['is_tree_edge'] = True 
                
                if not perf_mode:
                    curr_updates['es_tree']['edges'].append((v_data['tree_parent'],v,'MOD'))
            
            else:    
                v_data['tree_parent'] = -1
                orphans_Q = deque([v,])
                
                if not perf_mode:
                    v_data['original_orphan'] = True
                    self.refresh_update_dict(curr_updates)
                    yield(curr_updates, False)
                    curr_updates = self.get_new_update_dict()

                while len(orphans_Q) > 0:

                    curr_Q_node = orphans_Q.pop() 
                    curr_Q_node_data = self.es_graph.nodes[curr_Q_node]
                    curr_Q_node_data['level'] += 1 
                    
                    if not perf_mode:
                        curr_Q_node_data["curr_orphan"] = True 
                        curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))

                    if(curr_Q_node_data['level'] > self.es_graph.number_of_nodes()):

                        
                        
                        if not perf_mode:
                            self.es_delete_subgraph(curr_Q_node, curr_updates)
                            self.refresh_update_dict(curr_updates)
                            yield(curr_updates, True)
                            return
                        else:
                            self.es_delete_subgraph(curr_Q_node, None)
                            yield(None, True)
                            return


                    curr_Q_node_data['parents'] = curr_Q_node_data['friends'].copy() 
                    for former_friend in curr_Q_node_data['friends']: 
                        former_friend_data = self.es_graph.nodes[former_friend]
                        former_friend_data['friends'].remove(curr_Q_node)
                        former_friend_data['children'].add(curr_Q_node)
                        
                        if not perf_mode:
                            former_friend_data["curr_neighbor"] = True
                            curr_updates['es_tree']['nodes'].append((former_friend,'MOD'))
                            self.refresh_update_dict(curr_updates)
                            yield(curr_updates, True)
                            curr_updates = self.get_new_update_dict()
                            former_friend_data.pop("curr_neighbor")
                            curr_updates['es_tree']['nodes'].append((former_friend,'MOD'))
                            curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))

                    
                    curr_Q_node_data['friends'] = curr_Q_node_data['children'].copy()
                    for former_child in curr_Q_node_data['children']: 

                        former_child_data = self.es_graph.nodes[former_child]
                        former_child_data['parents'].remove(curr_Q_node)
                        former_child_data['friends'].add(curr_Q_node)

                        if not perf_mode:
                            former_child_data["curr_neighbor"] = True
                            curr_updates['es_tree']['nodes'].append((former_child,'MOD'))
                            curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))

                        if former_child_data['tree_parent'] == curr_Q_node:
                            self.es_graph.edges[(former_child_data['tree_parent'],former_child)]['is_tree_edge'] = False
                            if len(former_child_data['parents']) > 0:
                                former_child_data['tree_parent'] = former_child_data['parents'].pop()
                                former_child_data['parents'].add(former_child_data['tree_parent']) 
                                self.es_graph.edges[(former_child_data['tree_parent'],former_child)]['is_tree_edge'] = True
                            else:
                                former_child_data['tree_parent'] = -1
                                orphans_Q.appendleft(former_child) 

                            if not perf_mode:
                                curr_updates['es_tree']['edges'].append((former_child,curr_Q_node,'MOD'))
                        
                        if not perf_mode:
                            self.refresh_update_dict(curr_updates)
                            yield(curr_updates, False)
                            curr_updates = self.get_new_update_dict()
                            former_child_data.pop("curr_neighbor")
                            curr_updates['es_tree']['nodes'].append((former_child,'MOD'))
                        
                    curr_Q_node_data['children'] = set() 

                    if len(curr_Q_node_data['parents']) > 0:
                        curr_Q_node_data['tree_parent'] = curr_Q_node_data['parents'].pop()
                        curr_Q_node_data['parents'].add(curr_Q_node_data['tree_parent']) 
                        self.es_graph.edges[(curr_Q_node_data['tree_parent'],curr_Q_node)]['is_tree_edge'] = True
                        if not perf_mode:
                            curr_updates['es_tree']['edges'].append((curr_Q_node_data['tree_parent'],curr_Q_node,'MOD'))

                    else:
                        orphans_Q.append(curr_Q_node)

                    if not perf_mode:
                        yield(curr_updates, False)
                        curr_Q_node_data.pop("curr_orphan")
                        curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))

    
        if not perf_mode:
            self.refresh_update_dict(curr_updates)
            yield(curr_updates, True)
            return 
        
        else:
            yield(None, True)
            return 


# class ESVis(DynVis):

#     def __init__(self, algo:ESAlgo,fig:dict[str:go.Figure]):
#         figs_dict = {"es_tree":fig}
#         super().__init__(algo,figs_dict) 

#     def node_inf_to_hovertext(self,node_data):
#         return '<br>'.join([
#             f"Lvl: {node_data['level']}",
#             f"P_T: {node_data['tree_parent'] if node_data['tree_parent'] != -1 else 'None'}",
#             f"P: {node_data['parents']   if len(node_data['parents']) > 0 else 'None'}",
#             f"F: {node_data['friends']   if len(node_data['friends']) > 0 else 'None'}",
#             f"C: {node_data['children']  if len(node_data['children']) > 0 else 'None'}",
#         ])
#     def default_init_node_visdict(self, node: int, key: str):     
#         assert key == 'es_tree'

#         nx_graph = self.graphs_dict[key] 
#         node_data = nx_graph.nodes[node]

#         node_data['color'] = 'green' if node == self.algo_nx.start_node else 'red'
#         node_data['size'] = DEFAULT_NODE_SIZE
#         node_data['text_size'] = DEFAULT_TEXT_SIZE
#         node_data['hoverinfo'] = None 
#         node_data['hovertext'] = self.node_inf_to_hovertext(node_data)      
#         node_data['name'] = str(node)

#     def default_init_edge_visdict(self, u:int, v:int , key: str):
#         assert key == 'es_tree'
#         nx_graph = self.graphs_dict[key]
#         u,v = min(u,v), max(u,v)
#         edge = (u,v)
#         edge_data = nx_graph.edges[edge]

#         edge_data['name'] = f"{u}_{v}"
#         edge_data['color'] = 'red' if edge_data['is_tree_edge'] else DEFAULT_EDGE_COLOR
#         edge_data['width'] = DEFAULT_EDGE_WIDTH
#         edge_data['hoverinfo'] = 'none'
#         edge_data['hovertext'] = None

#     def default_init_nx_layout(self, key: str) -> None:
#         assert key == 'es_tree'
#         nx_graph = self.graphs_dict[key]
#         nx_positions = nx.bfs_layout(nx_graph, self.algo_nx.start_node,align='horizontal')

#         for node in nx_graph.nodes:
#             nx_positions[node][1] = -nx_positions[node][1]
#             nx_graph.nodes[node]['pos'] = nx_positions[node].tolist()

# if __name__ == "__main__": 
#     from pprint import pprint
#     base_graph = nx.Graph() 
#     base_graph.add_nodes_from(list(range(9))) 
#     base_graph.add_edges_from([
#         (0, 1),
#         (0, 2),
#         (0, 3),
#         (1, 2),
#         (1, 4),
#         (1, 5),
#         (1, 6),
#         (1, 7),
#         (2, 6),
#         (2, 7),
#         (3, 6),
#         (3, 7),
#         (4, 5),
#         (4, 8),
#         (5, 8),
#         (6, 8),
#         (7, 8)
#     ])

#     base_graph = nx.random_geometric_graph(60,0.15)
#     base_graph = base_graph.subgraph(nx.node_connected_component(base_graph,0)) 
#     bfs_algo = ESAlgo(base_graph, 0)
#     fig = new_default_fig()
#     fig.update_layout(hoverlabel=dict(font_size=18))
#     fig.update_layout(width=1500,height=900)
#     bfs_vis = ESVis(bfs_algo,fig)
#     bfs_vis.vis_init_all()

#     app = Dash(__name__)

#     app.layout = html.Div([
#         dcc.Graph(figure=fig)
#     ])
#     app.run_server(debug=True)