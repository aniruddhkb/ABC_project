'''
Implements the Even Shiloach edge-decremental single-source-shortest-path tree.

TODO: Convert to multi-source (But not all-pairs) version. Analogous to multi-source BFS.

Just have to make a few modifications.
'''

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

VIS_Y_EPSILON = 0.1
VIS_X_ORPH_COEFF = 0.1
VIS_X_DRIFT_COEFF = 0.8

class ESAlgov2(BFSAlgo,DynAlgo):
    '''
    modified ESAlgo to support multi-root and other systems.
    '''
    def __init__(self, base_graph:nx.Graph, start_node_arg:int|list[int],max_level:int|None = None, allowed_node_set:None|set = None):
        BFSAlgo.__init__(self,base_graph,start_node_arg,max_level,allowed_node_set) 
        DynAlgo.__init__(self,self.bfs_graph)
        self.es_graph = self.bfs_graph
        self.all_graphs['es_tree'] = self.es_graph
        keys = list(self.all_graphs.keys())
        for key in keys:
            if key != 'es_tree':
                self.all_graphs.pop(key)
        self.orphans = []

    def path_to_source(self, node:int):
        path = [node]
        while  not self.es_graph.nodes[node]['is_root']:
            node = self.es_graph.nodes[node]['tree_parent']
            path.append(node)
        return path
    
    def es_delete_subgraph(self, node:int, curr_updates:dict|None):
       
        nodes_del_Q = deque([node,]) 
        while len(nodes_del_Q) > 0:
            curr_node = nodes_del_Q.pop()
            self.shifted.append(curr_node)
            curr_node_level = self.es_graph.nodes[curr_node]['level'] 
            self.levels_to_nodes[curr_node_level].remove(curr_node)
            if(len(self.levels_to_nodes[curr_node_level]) == 0):
                self.levels_to_nodes.pop(curr_node_level)
            
            if not curr_updates is None:
                curr_updates['es_tree']['nodes'].append((curr_node,'DEL'))
            
            for neighbor in self.es_graph.neighbors(curr_node):
            
                if not curr_updates is None:
                    curr_updates['es_tree']['edges'].append(((curr_node,neighbor),'DEL')) 
            
                nodes_del_Q.appendleft(neighbor)
            self.es_graph.remove_node(curr_node)

    def es_delete_oneshot(self, u:int, v:int):
        return next(self.es_delete_genf(u,v))

    def es_delete_genf(self, u:int, v:int, perf_mode:bool=False):     
        
        self.shifted = []
        self.orphans = []
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
        elif u_data['level'] == v_data['level'] - 1 and v_data['tree_parent'] == u :
            assert u in v_data['parents'] and v in u_data['children']
            u_data['children'].remove(v)
            v_data['parents'].remove(u)

            if len(v_data['parents']) > 0:
                v_data['tree_parent'] = next(iter(v_data['parents']))
                self.es_graph.edges[(v_data['tree_parent'],v)]['is_tree_edge'] = True 
                v_data['root'] = self.es_graph.nodes[v_data['tree_parent']]['root']
                
                if not perf_mode:
                    curr_updates['es_tree']['edges'].append(((v_data['tree_parent'],v),'MOD'))

        ########## NONTRIVIAL CASE BELOW ##############
            else:    
                v_data['tree_parent'] = -1
                v_data['root'] = -1 
                
                orphans_Q = deque([v,])
                

                if not perf_mode:
                    self.orphans = []
                    self.orphans.append(v)
                    v_data['original_orphan'] = True
                    self.refresh_update_dict(curr_updates)
                    #print("STARTING ORPHANS Q FOR", v)
                    yield(curr_updates, False)
                    curr_updates = self.get_new_update_dict()
                
                while len(orphans_Q) > 0:  
                    curr_Q_node = orphans_Q.popleft()   
                    self.shifted.append(curr_Q_node)
                    curr_Q_node_data = self.es_graph.nodes[curr_Q_node]
                    curr_Q_node_data['curr_orphan'] = True
                    #print("POPPED FROM Q:", curr_Q_node)
                    #print(curr_Q_node_data)
                    if curr_Q_node_data['level'] == self.max_level:
                        #print("DELETING CURR Q NODE:", curr_Q_node)
                        for friend in self.es_graph.nodes[curr_Q_node]['friends']:
                            self.es_graph.nodes[friend]['friends'].remove(curr_Q_node)
                            self.es_graph.remove_edge(curr_Q_node,friend)
                            if not perf_mode:
                                curr_updates['es_tree']['edges'].append(((curr_Q_node,friend),'DEL'))
                                self.orphans.remove(curr_Q_node)
                        
                        self.levels_to_nodes[curr_Q_node_data['level']].remove(curr_Q_node)
                        self.es_graph.remove_node(curr_Q_node)
                        if not perf_mode:
                            curr_updates['es_tree']['nodes'].append((curr_Q_node,'DEL'))
                            self.refresh_update_dict(curr_updates)
                            yield(curr_updates, False)
                            curr_updates = self.get_new_update_dict()
                        continue

                            


                    if not perf_mode:
                        curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))
                        curr_Q_node_data["curr_orphan"] = True 
                        # self.refresh_update_dict(curr_updates)
                        # yield(curr_updates, False)
                        # curr_updates = self.get_new_update_dict()

                    self.levels_to_nodes[curr_Q_node_data['level']].remove(curr_Q_node)
                    curr_Q_node_data['level'] += 1 

                    if curr_Q_node_data['level'] not in self.levels_to_nodes:
                        self.levels_to_nodes[curr_Q_node_data['level']] = [] 
                    self.levels_to_nodes[curr_Q_node_data['level']].append(curr_Q_node)

                    if not perf_mode:
                        curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))
                        # self.refresh_update_dict(curr_updates)
                        # yield(curr_updates, False)
                        # curr_updates = self.get_new_update_dict()
                    
                    #print("POPPED FROM Q:", curr_Q_node)
                    
                    # DISCONNECTION CONDITION 2
                    if(len(self.levels_to_nodes[curr_Q_node_data['level'] - 1]) == 0):
                        self.levels_to_nodes.pop(curr_Q_node_data['level'] - 1)
                    
                        if not perf_mode:
                            self.es_delete_subgraph(curr_Q_node, curr_updates)
                            self.refresh_update_dict(curr_updates)
                            #print("A NODE WENT BELOW THE LINE:", curr_Q_node)
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
                    curr_Q_node_data['children'] = set()
                    for former_child in curr_Q_node_data['friends']:
                        former_child_data = self.es_graph.nodes[former_child]
                        former_child_data['parents'].remove(curr_Q_node)
                        former_child_data['friends'].add(curr_Q_node)
                        if former_child_data['tree_parent'] == curr_Q_node:
                            self.es_graph.edges[(former_child_data['tree_parent'],former_child)]['is_tree_edge'] = False
                            if not perf_mode:
                                curr_updates['es_tree']['edges'].append(((curr_Q_node,former_child),'MOD'))
                        
                        if not perf_mode:
                                curr_updates['es_tree']['nodes'].append((former_child,'MOD'))

                    for former_child in curr_Q_node_data['friends']: 
                        former_child_data = self.es_graph.nodes[former_child]  
                        if len(former_child_data['parents']) > 0:
                            former_child_data['tree_parent'] = former_child_data['parents'].pop()
                            former_child_data['parents'].add(former_child_data['tree_parent']) 

                            self.es_graph.edges[(former_child_data['tree_parent'],former_child)]['is_tree_edge'] = True
                            if not perf_mode:
                                curr_updates['es_tree']['edges'].append(((former_child,former_child_data['tree_parent']),'MOD'))
                    
                        else:
                            former_child_data['tree_parent'] = -1
                            orphans_Q.append(former_child) 
                            self.orphans.append(former_child)

                            if not perf_mode:
                                curr_updates['es_tree']['nodes'].append((former_child,'MOD'))
                    # if not perf_mode:
                    #     self.refresh_update_dict(curr_updates)
                    #     yield(curr_updates, False)
                    #     curr_updates = self.get_new_update_dict()
                    curr_Q_node_data['children'] = set() 
                    
                              

                    if len(curr_Q_node_data['parents']) > 0:
                        curr_Q_node_data['tree_parent'] = next(iter(curr_Q_node_data['parents']))
                        
                        curr_Q_node_data['root']=self.es_graph.nodes[curr_Q_node_data['tree_parent']]['root']

                        self.es_graph.edges[(curr_Q_node_data['tree_parent'],curr_Q_node)]['is_tree_edge'] = True                    
                        if not perf_mode:
                            #print("FINISHED AN ITERATION IN THE Q")
                            curr_updates['es_tree']['edges'].append(((curr_Q_node_data['tree_parent'],curr_Q_node),'MOD'))
                            curr_updates['es_tree']['nodes'].append((curr_Q_node,'MOD'))
                            self.refresh_update_dict(curr_updates)
                            yield(curr_updates, False)
                            curr_updates = self.get_new_update_dict()
                            self.orphans.remove(curr_Q_node)
                            curr_Q_node_data.pop("curr_orphan")
                    else:
                        orphans_Q.append(curr_Q_node)
                        #print("CURR_ORPHAN_STILL_ORPHANED:", curr_Q_node, orphans_Q)
                        if not perf_mode:
                            self.refresh_update_dict(curr_updates)
                            yield(curr_updates, False)
                            curr_updates = self.get_new_update_dict()

                if not perf_mode:
                    v_data.pop('original_orphan')
                    #print("FINISHED ORPHAN Q")
        #print("FINISHED DELETION")
        #print("CURRENT NODES:", self.es_graph.nodes)
        if not perf_mode:
            for node in self.es_graph.nodes:
                curr_updates['es_tree']['nodes'].append((node,'MOD'))
            for edge in self.es_graph.edges:
                curr_updates['es_tree']['nodes'].append((node,'MOD'))
            self.refresh_update_dict(curr_updates)
            yield(curr_updates, True)
            return 
        
        else:
            yield(None, True)
            return 
    
    def get_level(self,node:int)->int:
        assert node in self.es_graph.nodes
        return self.es_graph.nodes[node]['level']
    
    def get_root(self,node:int)->int:
        assert node in self.es_graph.nodes
        return self.es_graph.nodes[node]['root']
    
class ESVisv2(BFSVis,DynVis):

    def __init__(self, algo:ESAlgov2,fig:go.Figure):
        
        figs_dict = {"es_tree": fig}  
        BFSVis.__init__(self,algo,figs_dict)
        DynVis.__init__(self,algo,figs_dict)
        self.algo_nx = algo

    def default_init_node_visdict(self, node: int, key: str):     
        
        
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
        BFSVis.default_init_nx_layout(self,key)
  
    def lvl_awr_refresh_graphtrace_pos(self, key:str): 
        if(len(self.algo_nx.orphans) > 0):
             
            nx_graph = self.algo_nx.all_graphs[key] 
            
            shadow_edges = []
            for orphan in self.algo_nx.orphans: 
                orphan_level = nx_graph.nodes[orphan]['level'] 
                orphan_x = nx_graph.nodes[orphan]['pos'][0]

                candidate_shadow_parents = self.algo_nx.levels_to_nodes[orphan_level - 1] 
                shadow_parent = min(candidate_shadow_parents,key=lambda x: abs(nx_graph.nodes[x]['pos'][0] - orphan_x)) 
                if (shadow_parent,orphan) not in nx_graph.edges:
                    shadow_edges.append((shadow_parent,orphan))
            if len(shadow_edges) > 0:
                for shadow_edge in shadow_edges:
                    nx_graph.add_edge(*shadow_edge) 
                    nx_graph.edges[shadow_edge]['SHADOW'] = True

                alter_nodes = []
                [alter_nodes.extend(edge) for edge in shadow_edges]
                alter_nodes = set(alter_nodes)
                nx_layout = BFSVis.default_get_nx_layout(self,key)
                root_x_es = [nx_graph.nodes[node]['pos'][0] for node in self.algo_nx.multi_roots ]
                mean_root_x = sum(root_x_es)/len(root_x_es)
                root_x = mean_root_x
                for node in nx_graph.nodes:
                    old_x = nx_graph.nodes[node]['pos'][0]
                    nx_graph.nodes[node]["pos"][1] = -nx_layout[node][1]
                    if node not in self.algo_nx.orphans:
                        nx_graph.nodes[node]['pos'][0] = root_x + (nx_layout[node][0] - root_x)*(1-VIS_X_DRIFT_COEFF) + (old_x - root_x)*VIS_X_DRIFT_COEFF
                    else:
                        nx_graph.nodes[node]['pos'][0] = root_x + (nx_layout[node][0] - root_x)*(1-VIS_X_ORPH_COEFF) + (old_x - root_x)*VIS_X_ORPH_COEFF

                u,v = shadow_edges[0]

                delta_y = abs(nx_graph.nodes[v]['pos'][1] - nx_graph.nodes[u]['pos'][1])              

                for node in nx_graph.nodes:
                    y_orig = nx_graph.nodes[node]['pos'][1]
                    y_new = y_orig + random.uniform(-VIS_Y_EPSILON,VIS_Y_EPSILON) * delta_y
                    nx_graph.nodes[node]['pos'][1] = y_new 
                
                for shadow_edge in shadow_edges:
                    nx_graph.remove_edge(*shadow_edge)
            
            for node in nx_graph.nodes:
                self.vis_update_nodetrace(key,node)

            for edge in nx_graph.edges:
                    self.vis_update_edgetrace(key,*edge)

    def lvl_awr_vis_step_fn(self,all:bool=False):
        if all:
            step_result = self.algo_nx.step_all_remaining()
        else:
            step_result = self.algo_nx.step()
        
        to_update = step_result[0]
        for key in to_update.keys():
            nx_graph = self.algo_nx.all_graphs[key]
            to_update_subdict = to_update[key] 

            for(edge, opkeyword) in to_update_subdict['edges']:

                if opkeyword == 'DEL':
                    self.vis_delete_edgetrace(*edge,key)

                elif opkeyword == 'MOD':
                    if(edge in self.algo_nx.all_graphs[key].edges):
                        edge_data = self.algo_nx.all_graphs[key].edges[edge]
                        self.default_init_edge_visdict(*edge,key)
                        self.vis_update_edgetrace(key,*edge)

            for (node,opkeyword) in to_update_subdict['nodes']:

                if opkeyword == 'DEL':
                    self.vis_delete_nodetrace(node,key)
                
                elif opkeyword == 'MOD':
                    if(node in self.algo_nx.all_graphs[key].nodes):
                        node_data = self.algo_nx.all_graphs[key].nodes[node]
                        if("curr_orphan" in node_data):
                            curr_level = node_data['level'] 
                            assert curr_level - 1 in self.algo_nx.levels_to_nodes 
                            curr_x = node_data['pos'][0]

                            shadow_parents = self.algo_nx.levels_to_nodes[curr_level - 1]
                            shadow_delta_x_es = []
                            for shadow_parent in shadow_parents:
                                shadow_delta_x_es.append(abs(nx_graph.nodes[shadow_parent]['pos'][0] - curr_x ))
                            shadow_parent = shadow_parents[shadow_delta_x_es.index(min(shadow_delta_x_es))] 
                            shadow_y = nx_graph.nodes[shadow_parent]['pos'][1]
                            delta_y = abs(shadow_y - node_data['pos'][1])
                            shadow_edge = (shadow_parent,node) 
                            if shadow_edge not in self.algo_nx.all_graphs[key].edges:
                                self.algo_nx.all_graphs[key].add_edge(*shadow_edge)
                                new_layout = BFSVis.default_get_nx_layout(self,key)
                                node_data['pos'][1] = -new_layout[node][1] + random.uniform(-VIS_Y_EPSILON,VIS_Y_EPSILON) * delta_y
                                node_data['pos'][0] = new_layout[node][0] 
                                self.vis_update_nodetrace(key,node) 
                                self.algo_nx.all_graphs[key].remove_edge(*shadow_edge)
                                for neighbor in self.algo_nx.all_graphs[key].neighbors(node):
                                    self.vis_update_edgetrace(key,node,neighbor)

                            
                        self.default_init_node_visdict(node,key)
                        self.vis_update_nodetrace(key,node)

        if(len(self.algo_nx.orphans) > 0):
            self.lvl_awr_refresh_graphtrace_pos('es_tree')
        
        if step_result[1]:
            self.default_init_nx_layout('es_tree')
            for node in self.algo_nx.all_graphs['es_tree'].nodes:
                self.vis_update_nodetrace('es_tree',node)
            for edge in self.algo_nx.all_graphs['es_tree'].edges:
                self.vis_update_edgetrace('es_tree',edge[0],edge[1])
        return self.figs_dict['es_tree']
 

if __name__ == "__main__": 
    
    main_graph = nx.circulant_graph(9,[1]*9)
    # main_graph = get_connected_gnp_graph(50,25,0.1)
    
    # main_graph.add_edge(0,1)
    # main_graph.add_edge(1,2)

    es_algo = ESAlgov2(main_graph, (7,6),5) 
    es_algo.assign_generator(lambda: es_algo.es_delete_genf(0,1,True),)
    next(es_algo.update_genner)
    #print(es_algo.es_graph.nodes)
    fig = default_new_fig()
    es_vis = ESVisv2(es_algo,fig)
    es_vis.vis_init_all()
    fig.show()
    # fig.update_layout(hoverlabel=dict(font_size=18))
    # fig.update_layout(width=1500,height=900)



    # es_vis = ESVisv2(ESAlgov2(main_graph, 0,6),fig)
    # es_vis.vis_init_all()
    

    
    

    
            

    # app = Dash(__name__)

    # @app.callback(
    #         Output('es_fig', 'figure',True),
    #         Input('step_button', 'n_clicks'),
    #         prevent_initial_call=True,
    #     )
    # def incremental_step_callback(n_clicks):
        
    #     if n_clicks is not None:
    #         if es_vis.algo_nx.update_genner is None:
    #             es_vis.algo_nx.assign_generator(lambda: es_vis.algo_nx.es_update_delete_edge(0,1))
    #         es_vis.lvl_awr_vis_step_fn()
            
    #     return es_vis.figs_dict['es_tree']

    # @app.callback(
    #     Output('es_fig', 'figure',True),
    #     Input('redraw_button', 'n_clicks'),
    #     prevent_initial_call=True,
    #     allow_duplicate=True
    #     )
    # def redraw_callback(n_clicks):
    #     es_vis.default_init_nx_layout('es_tree')
        
    #     for node in es_vis.algo_nx.es_graph.nodes:
    #         es_vis.vis_update_nodetrace('es_tree',node)
    #     for edge in es_vis.algo_nx.es_graph.edges:
    #         es_vis.vis_update_edgetrace('es_tree',edge[0],edge[1])
    #     return es_vis.figs_dict['es_tree']

    # @app.callback(
    #     Output('es_fig', 'figure',True),
    #     Input('nx_reset_button', 'n_clicks'),
    #     prevent_initial_call=True,
    #     allow_duplicate=True
    #     )
    # def nx_reset_callback(n_clicks):
    #     fig.data = []
    #     es_vis.__init__(ESAlgov2(main_graph, 0,6),fig)
    #     es_vis.vis_init_all()
    #     return es_vis.figs_dict['es_tree']

    # app.layout = html.Div([
    #     html.Button('Next Step', id='step_button', n_clicks=None),
    #     html.Button('Redraw Tree', id='redraw_button', n_clicks=None),
    #     html.Button('Reset Graph', id='nx_reset_button', n_clicks=None),
    #     dcc.Graph(figure=fig, id = 'es_fig'),
    # ])
    # app.run_server(debug=True)