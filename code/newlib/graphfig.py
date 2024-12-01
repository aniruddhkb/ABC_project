
import networkx as nx 
from dash import Dash, html, dcc, Input, Output, callback, Patch
import plotly.graph_objects as go 
DEFAULT_TEXT_SIZE = 24
DEFAULT_NODE_SIZE = 10
DEFAULT_NODE_COLOR = 'red'
DEFAULT_EDGE_WIDTH = 2
DEFAULT_EDGE_COLOR = 'blue'
DEFAULT_FIG_SIZE = (1366,768)

class StatAlgo():
    def __init__(self, base_graph:nx.Graph):
        assert nx.number_of_selfloops(base_graph) == 0, 'Graph cannot have self loops'
        self.base_graph = base_graph.copy()
        self.all_graphs:dict[str,nx.Graph] = {}
        self.all_graphs['base'] = self.base_graph

class DynAlgo(StatAlgo):
    
    def __init__(self, base_graph):
        super().__init__(base_graph) 
        self.update_genner = None 

    def make_update_generator(self,is_add, edge):
        assert self.update_genner is None, 'Update generator already exists'
        if not isinstance(edge, tuple) or len(edge) != 2 or not isinstance(edge[0],int) or not isinstance(edge[1],int):
            raise ValueError('Edge must be a tuple of two integers')
        
        self.updated_nodes = []
        self.updated_edges = []
        
        if(is_add):
            assert edge not in self.base_graph.edges, 'Edge already exists'
        else:
            assert edge in self.base_graph.edges, 'Edge does not exist'

        self.update_genner = self.update_generator_function(is_add, edge)
    
    def step_all_remaining(self):
        while self.step():
            pass
        return self.updated_nodes, self.updated_edges
    
    def step(self):
        assert not self.update_genner is None, 'Update generator does not exist'

        try:
            return next(self.update_genner)
        except StopIteration:
            self.update_genner = None
            return False

    def update_generator_function(self, is_add:bool,edge:tuple[int,int]):
        if is_add:
            self.base_graph.add_edge(*edge)
        else:
            self.base_graph.remove_edge(*edge) 

        curr_updated_nodes = list(edge)
        curr_updated_edges = [edge,]
        
        self.updated_nodes.append(curr_updated_nodes)
        self.updated_edges.append(curr_updated_edges)  

        yield (curr_updated_nodes, curr_updated_edges)
        
        return True
        
class StatAlgoVis:

    def __init__(self, static_algo_nx:StatAlgo, figs_dict:dict[str,go.Figure]):
        
        self.static_algo_nx = static_algo_nx
        self.figs_dict = figs_dict
        self.graphs_dict = self.static_algo_nx.all_graphs
        assert [key in self.graphs_dict for key in self.figs_dict], 'All keys in figs_dict must be in graphs_dict'
        self.keys = self.figs_dict.keys()

    def default_init_all_figs(self):
        for key in self.keys:
            self.default_init_visdicts(key)
            self.default_init_nx_layout(key)
            self.default_init_add_traces_to_fig(key)

            
    def default_init_visdicts(self, key:str):
        
        nx_graph = self.graphs_dict[key] 
        for node in nx_graph.nodes:
            nx_graph.nodes[node]['color'] = DEFAULT_NODE_COLOR
            nx_graph.nodes[node]['size'] = DEFAULT_NODE_SIZE
            nx_graph.nodes[node]['text_size'] = DEFAULT_TEXT_SIZE
            nx_graph.nodes[node]['hoverinfo'] = None
            nx_graph.nodes[node]['hovertext'] = None
            nx_graph.nodes[node]['name'] = str(node)
        for edge in nx_graph.edges:
            edge_data = nx_graph.edges[edge]
            edge_data['name'] = f'{edge[0]}_{edge[1]}'
            edge_data['color'] = DEFAULT_EDGE_COLOR
            edge_data['width'] = DEFAULT_EDGE_WIDTH

    def default_init_nx_layout(self,key:str,)->None:
        nx_graph = self.graphs_dict[key]
        nx_positions = nx.spring_layout(nx_graph) 
        for node in nx_graph.nodes:
            nx_graph.nodes[node]['pos'] = nx_positions[node].tolist()
    
    def default_nxdict_to_node_trace(self, node_data:dict):
        x,y = node_data['pos']
        return go.Scatter(
            x=[x,], y=[y,],
            mode='markers+text',
            text=node_data['name'],
            textfont=dict(size=node_data['text_size']),
            name=node_data['name'],
            textposition='top right',
            customdata=[node_data['hovertext'],],
            hoverinfo=node_data["hoverinfo"],
            hovertemplate='%{customdata}', 
            marker=go.scatter.Marker(color=node_data["color"],size=node_data['size'])
            )
    
    def default_nxdict_to_edge_trace(self, edge:tuple[int,int], nx_graph:nx.Graph):
        u,v = edge
        edge_data = nx_graph.edges[edge]
        u_dt,v_dt = nx_graph.nodes[u],nx_graph.nodes[v]
        u_x,u_y = u_dt['pos']
        v_x,v_y = v_dt['pos']
        return go.Scatter(
                x=(u_x,v_x),
                y=(u_y,v_y),
                mode='lines',
                name=edge_data['name'],
                line=dict(width=edge_data['width'], color=edge_data['color']),
                hoverinfo='none',
            )

    def default_init_add_traces_to_fig(self, key:str):
        nx_graph = self.graphs_dict[key] 
        fig = self.figs_dict[key] 

        for node in nx_graph.nodes:
            node_data = nx_graph.nodes[node]
            fig.add_trace(self.default_nxdict_to_node_trace(node_data)) 
            new_trace = fig.data[-1] 
            assert new_trace['name'] == node_data['name'] 
            node_data['trace'] = new_trace

        for edge in nx_graph.edges:
            edge_data = nx_graph.edges[edge]
            fig.add_trace(self.default_nxdict_to_edge_trace(edge,nx_graph))
            new_trace = fig.data[-1]
            assert new_trace['name'] == edge_data['name']
            edge_data['trace'] = new_trace
    

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
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from([1,2,3,4,5])
    nx_graph.add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,1)])
    figs_dict = {
        'base':default_new_fig(),
    }
    static_algo_nx = StatAlgo(nx_graph)
    nx_graph = static_algo_nx.all_graphs['base']
    static_algo_vis = StatAlgoVis(static_algo_nx, figs_dict)
    static_algo_vis.default_init_all_figs()
    
    # for node in nx_graph.nodes:
    #     print(nx_graph.nodes[node])

    # for edge in nx_graph.edges:
    #     print(nx_graph.edges[edge])

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(figure=figs_dict['base'])
    ])
    app.run_server(debug=True)