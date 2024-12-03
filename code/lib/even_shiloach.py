from graphfig import * 
from bfs import *
import networkx as nx
from collections import deque
import random 
random.seed(31415)


DEFAULT_TEXT_SIZE = 24
DEFAULT_NODE_SIZE = 20

DEFAULT_EDGE_WIDTH = 4
DEFAULT_EDGE_COLOR = '#555555'
DEFAULT_TREE_EDGE_COLOR = '#AA0000'

DEFAULT_FIG_SIZE = (1280,600)


DEFAULT_NODE_COLOR = DEFAULT_EDGE_COLOR
ROOT_NODE_COLOR = DEFAULT_NODE_COLOR
OG_ORPHAN_COLOR = 'red'
ORPHAN_NODE_COLOR = 'blue'
CURR_ORPHAN_COLOR = 'green'
NEIGHBORS_COLOR = 'blue'
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
                'children' : set(),
                'is_root' : node == self.start_node
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
                    curr_updates['es_tree']['edges'].append(((curr_node,neighbor),'DEL')) 
            
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
            curr_updates['es_tree']['edges'].append(((u,v),'DEL'))
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
                    curr_updates['es_tree']['edges'].append(((v_data['tree_parent'],v),'MOD'))
            
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
                        self.refresh_update_dict(curr_updates)
                        yield(curr_updates, False) 
                        curr_updates = self.get_new_update_dict()

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
                            
                            curr_updates['es_tree']['nodes'].append((former_friend,'MOD'))
                            

                    
                    curr_Q_node_data['friends'] = curr_Q_node_data['children'].copy()
                    for former_child in curr_Q_node_data['children']: 

                        former_child_data = self.es_graph.nodes[former_child]
                        former_child_data['parents'].remove(curr_Q_node)
                        former_child_data['friends'].add(curr_Q_node)

                        if not perf_mode:
                            curr_updates['es_tree']['nodes'].append((former_child,'MOD'))

                        if former_child_data['tree_parent'] == curr_Q_node:
                            self.es_graph.edges[(former_child_data['tree_parent'],former_child)]['is_tree_edge'] = False
                            if len(former_child_data['parents']) > 0:
                                former_child_data['tree_parent'] = former_child_data['parents'].pop()
                                former_child_data['parents'].add(former_child_data['tree_parent']) 
                                self.es_graph.edges[(former_child_data['tree_parent'],former_child)]['is_tree_edge'] = True

                                if not perf_mode:
                                    curr_updates['es_tree']['edges'].append(((former_child,former_child_data['tree_parent']),'MOD'))
                            else:
                                former_child_data['tree_parent'] = -1
                                orphans_Q.appendleft(former_child) 

                            if not perf_mode:
                                curr_updates['es_tree']['edges'].append(((former_child,curr_Q_node),'MOD'))
                        

                        
                    curr_Q_node_data['children'] = set() 

                    if len(curr_Q_node_data['parents']) > 0:
                        curr_Q_node_data['tree_parent'] = curr_Q_node_data['parents'].pop()
                        curr_Q_node_data['parents'].add(curr_Q_node_data['tree_parent']) 
                        self.es_graph.edges[(curr_Q_node_data['tree_parent'],curr_Q_node)]['is_tree_edge'] = True
                        
                        if not perf_mode:
                            curr_updates['es_tree']['edges'].append(((curr_Q_node_data['tree_parent'],curr_Q_node),'MOD'))

                    else:
                        orphans_Q.append(curr_Q_node)

                    if not perf_mode:
                        yield(curr_updates, False)
                        curr_Q_node_data.pop("curr_orphan")
                        curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))

                if not perf_mode:
                    v_data.pop('original_orphan')
    
        if not perf_mode:
            self.refresh_update_dict(curr_updates)
            yield(curr_updates, True)
            return 
        
        else:
            yield(None, True)
            return 


class ESVis(DynVis):

    def __init__(self, algo:ESAlgo,fig:go.Figure):
        figs_dict = {"es_tree":fig}
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
        
        assert key == 'es_tree'
        nx_graph = self.graphs_dict[key]
        node_data = nx_graph.nodes[node]

        node_data['color'] = DEFAULT_NODE_COLOR
        
        if(node_data['is_root']):
            node_data['color'] = ROOT_NODE_COLOR
        
        if(node_data['level'] != 0 and node_data['tree_parent'] == -1):
            node_data['color'] = ORPHAN_NODE_COLOR

        if('curr_orphan' in node_data):
            node_data['color'] = CURR_ORPHAN_COLOR
        
        if('original_orphan' in node_data):
            node_data['color'] = OG_ORPHAN_COLOR

        
        node_data['size'] = DEFAULT_NODE_SIZE
        node_data['text_size'] = DEFAULT_TEXT_SIZE
        node_data['hoverinfo'] = None 
        node_data['hovertext'] = self.node_inf_to_hovertext(node_data)      
        node_data['name'] = str(node)

    def default_init_edge_visdict(self, u:int, v:int , key: str):
        assert key == 'es_tree'
        nx_graph = self.graphs_dict[key]
        u,v = min(u,v), max(u,v)
        edge = (u,v)
        edge_data = nx_graph.edges[edge]

        edge_data['name'] = f"{u}_{v}"
        edge_data['color'] = DEFAULT_TREE_EDGE_COLOR if edge_data['is_tree_edge'] else DEFAULT_EDGE_COLOR
        edge_data['width'] = DEFAULT_EDGE_WIDTH
        edge_data['hoverinfo'] = 'none'
        edge_data['hovertext'] = None

    def default_init_nx_layout(self, key: str) -> None:
        assert key == 'es_tree'
        nx_graph = self.graphs_dict[key]
        nx_positions = nx.bfs_layout(nx_graph, self.algo_nx.start_node,align='horizontal')

        for node in nx_graph.nodes:
            nx_positions[node][1] = -nx_positions[node][1]
            nx_graph.nodes[node]['pos'] = nx_positions[node].tolist()

    
    def lvl_awr_refresh_nx_layout(self, key:str):
        assert key == 'es_tree'
        nx_graph = self.algo_nx.all_graphs[key]
        desired_levels_to_orphans = {}
        desired_levels_to_y = {}
        orphan_nodes = []
        for node in nx_graph.nodes:
            node_data = nx_graph.nodes[node]
            if(node_data['tree_parent'] == -1 and node_data['level'] != 0):
                orphan_nodes.append(node)
                if node_data['level'] not in desired_levels_to_orphans:
                    desired_levels_to_orphans[node_data['level']] = []
                desired_levels_to_orphans[node_data['level']].append(node) 
            
        for node in nx_graph.nodes:
            node_data = nx_graph.nodes[node]
        
            if node not in orphan_nodes and node_data['level'] in desired_levels_to_orphans:
                if node_data['level'] not in desired_levels_to_y:
                    print('settled level', node_data['level'],', node', node)
                    desired_levels_to_y[node_data['level']] = nx_graph.nodes[node]['pos'][1] 
        for level in desired_levels_to_orphans:
            level_orphans = desired_levels_to_orphans[level]
            lvl_y = desired_levels_to_y[level]
            
            for orphan in level_orphans:
                nx_graph.nodes[orphan]['pos'][1] = lvl_y
        
        print(desired_levels_to_orphans)

    def lvl_awr_vis_step_fn(self):
        self.skeleton_vis_step(False)
        self.lvl_awr_refresh_nx_layout('es_tree')
        for u in self.algo_nx.es_graph.nodes:
            
            if(self.traces_dict['es_tree'][str(u)]['y'] != self.algo_nx.es_graph.nodes[u]['pos'][1]):
                self.vis_update_nodetrace('es_tree',u)
                for v in self.algo_nx.es_graph.neighbors(u):

                    u1, v1 = min(u,v), max(u,v)
                    self.vis_update_edgetrace('es_tree',u1,v1)
        return self.figs_dict['es_tree']
            
    

if __name__ == "__main__": 
    
    base_graph = nx.circulant_graph(9,[1]*9)

    # base_graph = nx.Graph() 
    # base_graph.add_nodes_from(list(range(9))) 
    # base_graph.add_edges_from([
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (1, 2),
    #     (1, 4),
    #     (1, 5),
    #     (1, 6),
    #     (1, 7),
    #     (2, 6),
    #     (2, 7),
    #     (3, 6),
    #     (3, 7),
    #     (4, 5),
    #     (4, 8),
    #     (5, 8),
    #     (6, 8),
    #     (7, 8)
    # ])

    # base_graph = nx.random_geometric_graph(60,0.15)
    # base_graph = base_graph.subgraph(nx.node_connected_component(base_graph,0)) 

    fig = new_default_fig()
    fig.update_layout(hoverlabel=dict(font_size=18))
    fig.update_layout(width=1500,height=900)
    es_vis = ESVis(ESAlgo(base_graph, 0),fig)
    es_vis.vis_init_all()
    # fig.show()
    app = Dash(__name__)
    @app.callback(
            Output('es_fig', 'figure',True),
            Input('step_button', 'n_clicks'),
            prevent_initial_call=True,
            
        )
    def incremental_step_callback(n_clicks):
        
        if n_clicks is not None:
            if es_vis.algo_nx.update_genner is None:
                es_vis.algo_nx.assign_generator(lambda: es_vis.algo_nx.es_update_delete_edge(0,1))
            es_vis.lvl_awr_vis_step_fn()
            
        return es_vis.figs_dict['es_tree']

    @app.callback(
        Output('es_fig', 'figure',True),
        Input('redraw_button', 'n_clicks'),
        prevent_initial_call=True,
        allow_duplicate=True
        )
    def redraw_callback(n_clicks):
        es_vis.default_init_nx_layout('es_tree')
        
        for node in es_vis.algo_nx.es_graph.nodes:
            es_vis.vis_update_nodetrace('es_tree',node)
        for edge in es_vis.algo_nx.es_graph.edges:
            es_vis.vis_update_edgetrace('es_tree',edge[0],edge[1])
        return es_vis.figs_dict['es_tree']

    @app.callback(
        Output('es_fig', 'figure',True),
        Input('nx_reset_button', 'n_clicks'),
        prevent_initial_call=True,
        allow_duplicate=True
        )
    def nx_reset_callback(n_clicks):
        fig.data = []
        es_vis.__init__(ESAlgo(base_graph,0),fig)
        es_vis.vis_init_all()
        return es_vis.figs_dict['es_tree']



    app.layout = html.Div([
        html.Button('Next Step', id='step_button', n_clicks=None),
        html.Button('Redraw Tree', id='redraw_button', n_clicks=None),
        html.Button('Reset Graph', id='nx_reset_button', n_clicks=None),
        dcc.Graph(figure=fig, id = 'es_fig'),
    ])
    app.run_server(debug=True)