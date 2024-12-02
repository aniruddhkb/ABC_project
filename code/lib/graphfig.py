
from typing import Callable
from collections.abc import Generator
import networkx as nx 
from dash import Dash, html, dcc, Input, Output, callback, Patch
import plotly.graph_objects as go 
from typing import Type


DEFAULT_TEXT_SIZE = 24
DEFAULT_NODE_SIZE = 10
DEFAULT_NODE_COLOR = 'black'
ALT_NODE_COLOR = 'green'
DEFAULT_EDGE_WIDTH = 2
DEFAULT_EDGE_COLOR = 'blue'
ALT_EDGE_COLOR = 'orange'
DEFAULT_FIG_SIZE = (1366,768)

class StatAlgo():
    def __init__(self, base_graph:nx.Graph):
        assert nx.number_of_selfloops(base_graph) == 0, 'Graph cannot have self loops'
        self.base_graph = base_graph.copy()
        self.all_graphs:dict[str,nx.Graph] = {}
        self.all_graphs['base'] = self.base_graph
        self.keys = self.all_graphs.keys()


class DynAlgo(StatAlgo):
    
    def __init__(self, base_graph):
        super().__init__(base_graph) 
        self.update_genner = None 
        self.keys = self.all_graphs.keys()
        self.add_del_mod_dict = {"ADD":(True,False,False), "DEL":(False,True,False), "MOD":(False,False,True)}

    def get_new_update_dict(self):
        new_dict = {}
        for key in self.keys:
            new_dict[key] = {'nodes':[], 'edges':[]}
        return new_dict
        
    def nx_add_edge(self,key,*edge):
        self.all_graphs[key].add_edge(*edge) 

    def nx_remove_edge(self,key,*edge):
        self.all_graphs[key].remove_edge(*edge)

    def nx_add_node(self,key,node):
        self.all_graphs[key].add_node(node)
    
    def nx_remove_node(self,key,node):
        nx_graph = self.all_graphs[key]
        assert node in nx_graph.nodes, 'Node does not exist'
        edges = list(nx_graph.edges(node))
        if len(edges) > 0:
            raise ValueError('Node has edges. Remove edges first')
        self.all_graphs[key].remove_node(node)

    def assign_generator(self, update_fn:Callable[[],Generator]):
        assert self.update_genner is None, 'Creating new update before previous update finished. Forbidden as it may lead to graph inconsistencies.'
        self.update_genner = update_fn()
    
    def step_all_remaining(self):
        while True:
            out = self.step()
            if out[1]:
                break
        return self.update_dict, True
    
    def step(self):

        assert not self.update_genner is None, 'No update in progress. Use make_generator to start a new update.'
        out = next(self.update_genner)
        
        if out[1]:
            self.update_genner = None
        return out

    def refresh_update_dict(self, curr_updates:dict):
        for key in curr_updates.keys(): 
            self.update_dict[key]['nodes'].extend(curr_updates[key]['nodes'])
            self.update_dict[key]['edges'].extend(curr_updates[key]['edges'])

    def yieldtest_update_fn(self):
        
        assert len(self.base_graph.nodes) == 0, 'Graph must be empty'
        self.update_dict = self.get_new_update_dict()
        #Adding and modding nodes 

        for i in range(5): 
        
            curr_updates = self.get_new_update_dict()
            self.nx_add_node('base', i)
            curr_updates['base']['nodes'].append((i,'ADD'))
            self.refresh_update_dict(curr_updates)
            yield (curr_updates, False)

            curr_updates = self.get_new_update_dict()
            curr_updates['base']['nodes'].append((i,'MOD'))
            self.refresh_update_dict(curr_updates)
            yield (curr_updates, False) 
        
        #Adding and modding edges
        for i in range(5):
            for j in range(i):

                curr_updates = self.get_new_update_dict()
                self.nx_add_edge('base', i, j)
                curr_updates['base']['edges'].append(((i,j),'ADD'))
                self.refresh_update_dict(curr_updates)
                yield (curr_updates, False)

                curr_updates = self.get_new_update_dict()
                curr_updates['base']['edges'].append(((i,j),'MOD'))
                self.refresh_update_dict(curr_updates)
                yield (curr_updates, False)


        #Modding and deleting edges
        for i in range(5):
            for j in range(i):
                
                curr_updates = self.get_new_update_dict()
                curr_updates['base']['edges'].append(((i,j),'MOD'))
                self.refresh_update_dict(curr_updates)
                yield (curr_updates, False)

                curr_updates = self.get_new_update_dict()
                self.nx_remove_edge('base', i, j)
                curr_updates['base']['edges'].append(((i,j),'DEL'))
                self.refresh_update_dict(curr_updates)
                yield (curr_updates, False)
        
        #Modding and deleting nodes
        for i in range(5):
            curr_updates = self.get_new_update_dict()
            curr_updates['base']['nodes'].append((i,'MOD'))
            yield (curr_updates, False)
            curr_updates = self.get_new_update_dict()
            self.nx_remove_node('base', i)
            curr_updates['base']['nodes'].append((i,'DEL'))
            yield(curr_updates, i==4 )

    def example_update_fn(self, is_node:bool, add_del_or_mod:str, graphelem:int|tuple[int, int]):
        self.update_dict = self.get_new_update_dict()
        curr_updates = self.get_new_update_dict()
        is_add, is_del, is_mod = self.add_del_mod_dict[add_del_or_mod]
        
        if is_add:
            if is_node:
                self.nx_add_node('base', graphelem)
                curr_updates['base']['nodes'].append((graphelem,'ADD'))
                curr_updates['base']['nodes'].append((graphelem,'MOD'))
            else:
                self.nx_add_edge('base', *graphelem)
                curr_updates['base']['edges'].append((graphelem,'ADD'))
        elif is_del:
            if is_node:
                self.nx_remove_node('base', graphelem)
                curr_updates['base']['nodes'].append((graphelem,'DEL'))
            else:
                self.nx_remove_edge('base', *graphelem)
                curr_updates['base']['edges'].append((graphelem,'DEL'))
        elif is_mod:
            if is_node:
                curr_updates['base']['nodes'].append((graphelem,'MOD'))
            else:
                curr_updates['base']['edges'].append((graphelem,'MOD'))
        for key in self.keys:
            self.update_dict[key]['nodes'].extend(curr_updates[key]['nodes'])
            self.update_dict[key]['edges'].extend(curr_updates[key]['edges'])                
        yield (curr_updates, True)
        
class StatVis:

    def __init__(self, algo_nx: Type[StatAlgo], figs_dict:dict[str,go.Figure]):
        
        self.algo_nx = algo_nx
        self.figs_dict = figs_dict
        self.keys = self.figs_dict.keys()
        self.graphs_dict = self.algo_nx.all_graphs
        assert [key in self.graphs_dict for key in self.figs_dict], 'All keys in figs_dict must be in graphs_dict'
        
        self.traces_dict = {}
        for key in self.keys:
            self.traces_dict[key] = {}
    

    
    def default_init_edge_visdict(self, u:int,v:int, key:str):
        
        nx_graph = self.graphs_dict[key]
        u,v = min(u,v), max(u,v)
        edge = (u,v)

        edge_data = nx_graph.edges[edge]
        edge_data['name'] = f'{u}_{v}'
        edge_data['color'] = DEFAULT_EDGE_COLOR
        edge_data['width'] = DEFAULT_EDGE_WIDTH

    def default_init_node_visdict(self, node:int, key:str):
        nx_graph = self.graphs_dict[key] 
        node_data = nx_graph.nodes[node]
        node_data['color'] = DEFAULT_NODE_COLOR
        node_data['size'] = DEFAULT_NODE_SIZE
        node_data['text_size'] = DEFAULT_TEXT_SIZE
        node_data['hoverinfo'] = 'none'
        node_data['hovertext'] = None
        node_data['name'] = str(node)

    def default_init_nx_layout(self,key:str,)->None:
        nx_graph = self.graphs_dict[key]
        nx_positions = nx.spring_layout(nx_graph) 
        for node in nx_graph.nodes:
            nx_graph.nodes[node]['pos'] = nx_positions[node].tolist()
     
    def vis_add_traces(self, key:str):
        nx_graph = self.graphs_dict[key]
        for node in nx_graph.nodes:
            self.vis_add_node(key,node)
        for edge in nx_graph.edges:
            self.vis_add_edge(key,*edge)

    def vis_add_edge(self,key:str,u:int, v:int):
        edge = (min(u,v), max(u,v))
        u,v = edge
        nx_graph = self.graphs_dict[key]
        fig = self.figs_dict[key]
        traces_subdict = self.traces_dict[key]
        edge_data = nx_graph.edges[edge]
        u_dt,v_dt = nx_graph.nodes[u],nx_graph.nodes[v]
        u_x,u_y = u_dt['pos']
        v_x,v_y = v_dt['pos']
        fig.add_trace(
            go.Scatter(
                x=(u_x,v_x),
                y=(u_y,v_y),
                mode='lines',
                name=edge_data['name'],
                line=dict(width=edge_data['width'], color=edge_data['color']),
                hoverinfo='none',
            )
        )
        new_trace = fig.data[-1]
        assert new_trace['name'] == edge_data['name']
        traces_subdict[edge_data['name']] = new_trace
        
    def vis_add_node(self,key:str,node:int):
        nx_graph = self.graphs_dict[key]
        fig = self.figs_dict[key]
        traces_subdict = self.traces_dict[key]
        node_data = nx_graph.nodes[node]

        x,y = node_data['pos']
        fig.add_trace(go.Scatter(
            x=[x,], y=[y,],
            mode='markers+text',
            text=node_data['name'],
            textfont=dict(size=node_data['text_size']),
            name=node_data['name'],
            textposition='top right',
            customdata=[node_data['hovertext'],],
            hoverinfo=node_data["hoverinfo"],
            hovertemplate='%{customdata}' if node_data['hoverinfo'] != 'none' else None,
            marker=go.scatter.Marker(color=node_data["color"],size=node_data['size'])
            ))
        new_trace = fig.data[-1]
        assert new_trace['name'] == node_data['name']
        traces_subdict[str(node)] = new_trace

    def vis_init_visdicts(self, key:str):
        nx_graph = self.graphs_dict[key] 
        for node in nx_graph.nodes:
            self.default_init_node_visdict(node, key)
        for edge in nx_graph.edges:
            self.default_init_edge_visdict(edge[0],edge[1], key)

    def vis_init_all(self):
        for key in self.keys:
            self.vis_init_visdicts(key)
            self.default_init_nx_layout(key)
            self.vis_add_traces(key)

class DynVis(StatVis):

    def __init__(self, dyn_algo_nx:DynAlgo, figs_dict:dict[str,go.Figure]):
        super().__init__(dyn_algo_nx, figs_dict)
        assert isinstance(dyn_algo_nx, DynAlgo), 'algo_nx must be of type DynAlgo'
        self.algo_nx = dyn_algo_nx
        

    def vis_delete_edge(self,u:int,v:int,key:str):
        u,v = min(u,v), max(u,v)
        traces_subdict = self.traces_dict[key]
        assert f'{u}_{v}' in traces_subdict, 'Edge does not exist'
        traces_subdict[f'{u}_{v}']['visible'] = False
        traces_subdict.pop(f'{u}_{v}') 

    def vis_delete_node(self,node:int,key:str):
        traces_subdict = self.traces_dict[key]
        assert str(node) in traces_subdict, 'Node does not exist'
        traces_subdict[str(node)]['visible'] = False
        traces_subdict.pop(str(node))

    def vis_update_node(self,key:str,node:int):
        node_data = self.algo_nx.all_graphs[key].nodes[node] 
        trace = self.traces_dict[key][str(node)]
        x,y = node_data['pos']
        trace['x'] = [x,] 
        trace['y'] = [y,]
        trace['customdata'] = [node_data['hovertext'],] 
        trace['hoverinfo'] = node_data['hoverinfo'] 
        trace['hovertemplate'] = '%{customdata}' if node_data['hoverinfo'] != 'none' else None 
        trace['marker']['color'] = node_data['color']
        trace['marker']['size'] = node_data['size']
        trace['textfont']['size'] = node_data['text_size']

    def vis_update_edge(self,key:str,u,v):
        u,v = min(u,v), max(u,v)
        edge_data = self.algo_nx.all_graphs[key].edges[(u,v)]
        trace = self.traces_dict[key][f'{u}_{v}']
        u_x, u_y = self.algo_nx.all_graphs[key].nodes[u]['pos'] 
        v_x, v_y = self.algo_nx.all_graphs[key].nodes[v]['pos'] 
        trace['x'] = (u_x,v_x)
        trace['y'] = (u_y,v_y)
        trace['line']['color'] = edge_data['color']
        trace['line']['width'] = edge_data['width'] 

    def default_add_node_to_nx_layout(self, key:str, node:int):
        nx_graph = self.algo_nx.all_graphs[key]
        pos_dict = {}
        for node in nx_graph.nodes:
            if('pos' in nx_graph.nodes[node]):
                pos_dict[node] = nx_graph.nodes[node]['pos']
        if(node in pos_dict):
            pos_dict.pop(node)
        if(len(pos_dict.keys()) == 0):
            new_layout = nx.spring_layout(nx_graph)
        else:
            new_layout = nx.spring_layout(nx_graph, pos=pos_dict)
        nx_graph.nodes[node]['pos'] = new_layout[node].tolist()


    def skeleton_vis_step(self, all:bool=False):
        if all:
            step_result = self.algo_nx.step_all_remaining()
        else:
            step_result = self.algo_nx.step()
        
        to_update = step_result[0]
        
        for key in to_update.keys():
            to_update_subdict = to_update[key] 
            for (node,opkeyword) in to_update_subdict['nodes']:
                if opkeyword == 'ADD':
                    self.default_add_node_to_nx_layout(key,node)
                    self.default_init_node_visdict(node,key)
                    node_data = self.algo_nx.all_graphs[key].nodes[node]
                    
                    # ... and do something right here. Mod the visdict. If you need to.

                    self.vis_add_node(key,node)

                elif opkeyword == 'DEL':
                    self.vis_delete_node(node,key)
                
                elif opkeyword == 'MOD':
                    node_data = self.algo_nx.all_graphs[key].nodes[node]

                    # ... and do something right here. Mod the visdict. If you need to.

                    self.vis_update_node(key,node)

            for(edge, opkeyword) in to_update_subdict['edges']:

                if opkeyword == 'ADD':
                    self.default_init_edge_visdict(*edge,key)

                    # ... and do something right here. Mod the visdict. If you need to. 

                    self.vis_add_edge(key,*edge) 

                elif opkeyword == 'DEL':
                    self.vis_delete_edge(*edge,key)

                elif opkeyword == 'MOD':
                    edge_data = self.algo_nx.all_graphs[key].edges[edge]

                    # ... and do something right here. Mod the visdict. If you need to. 

                    self.vis_update_edge(key,*edge)

        return self.figs_dict[key]

    def example_vis_step(self, all:bool=False):
        if all:
            step_result = self.algo_nx.step_all_remaining()
        else:
            step_result = self.algo_nx.step()
        
        if step_result[1]:
            self.update_in_progress = False
        to_update = step_result[0]
        
        for key in to_update.keys():
            to_update_subdict = to_update[key] 
            for (node,opkeyword) in to_update_subdict['nodes']:
                
                if opkeyword == 'ADD':
                    self.default_add_node_to_nx_layout(key,node)
                    self.default_init_node_visdict(node,key)
                    self.vis_add_node(key,node)

                elif opkeyword == 'DEL':
                    self.vis_delete_node(node,key)
                
                elif opkeyword == 'MOD':
                    self.vis_update_node(key,node)

            for(edge, opkeyword) in to_update_subdict['edges']:
                if opkeyword == 'ADD':

                    self.default_init_edge_visdict(*edge,key)
                    self.vis_add_edge(key,*edge)
                elif opkeyword == 'DEL':
                    self.vis_delete_edge(*edge,key)
                elif opkeyword == 'MOD':
                    self.vis_update_edge(key,*edge)
        
        return self.figs_dict[key]

    def yieldtest_vis_step(self,all:bool=False):

        if all:
            step_result = self.algo_nx.step_all_remaining()
        else:
            step_result = self.algo_nx.step()
            
        
        to_update = step_result[0]
        
        for key in to_update.keys():
            to_update_subdict = to_update[key] 
            for (node,opkeyword) in to_update_subdict['nodes']:
                if opkeyword == 'ADD':
                    self.default_add_node_to_nx_layout(key,node)
                    self.default_init_node_visdict(node,key)
                    self.vis_add_node(key,node)
                elif opkeyword == 'DEL':
                    self.vis_delete_node(node,key)
                elif opkeyword == 'MOD':
                    node_data = self.algo_nx.all_graphs[key].nodes[node]
                    if(node_data['color'] == DEFAULT_NODE_COLOR):
                        node_data['color'] = ALT_NODE_COLOR
                    else: 
                        node_data['color'] = DEFAULT_NODE_COLOR
                    self.vis_update_node(key,node)
            
            for(edge, opkeyword) in to_update_subdict['edges']:
                if opkeyword == 'ADD':
                    self.default_init_edge_visdict(*edge,key)
                    self.vis_add_edge(key,*edge)
                elif opkeyword == 'DEL':
                    self.vis_delete_edge(*edge,key)
                elif opkeyword == 'MOD':
                    edge_data = self.algo_nx.all_graphs[key].edges[edge]
                    if(edge_data['color'] == DEFAULT_EDGE_COLOR):
                        edge_data['color'] = ALT_EDGE_COLOR
                    else: 
                        edge_data['color'] = DEFAULT_EDGE_COLOR
                    self.vis_update_edge(key,*edge)
        return self.figs_dict[key]

def default_new_fig():
    fig = go.Figure(
        data=None,
        layout=go.Layout(
            title=dict(
                text='Graph',
                font=dict(
                    size=DEFAULT_TEXT_SIZE,
                )
            ),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=DEFAULT_FIG_SIZE[0],
            height=DEFAULT_FIG_SIZE[1],
        )
    )
    return fig

if __name__ == '__main__':
    app = Dash(__name__)

    nx_graph = nx.Graph()
    figs_dict = {
        'base':default_new_fig(),
    }
    dyn_vis = DynVis(DynAlgo(nx_graph), figs_dict)
    dyn_vis.vis_init_all()

    @app.callback(
        Output('base_fig', 'figure'),
        Input('step_button', 'n_clicks'),
        suppress_initial_call=True
    )
    def incremental_step_callback(n_clicks):
        
        if n_clicks is not None:
            if dyn_vis.algo_nx.update_genner is None:
        
                dyn_vis.algo_nx.assign_generator(dyn_vis.algo_nx.yieldtest_update_fn)
        
        
            dyn_vis.yieldtest_vis_step()
        return dyn_vis.figs_dict['base']
    
    app.layout = html.Div([
        html.Button('Incremental Step', id='step_button'),
        dcc.Graph(figure=figs_dict['base'], id='base_fig')
    ])
    app.run_server(debug=True)
