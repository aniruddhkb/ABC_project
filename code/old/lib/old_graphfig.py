from collections.abc import Generator
import networkx as nx 
import queue
from dash import Dash, html, dcc, Input, Output, callback, Patch
import plotly.graph_objects as go 
DEFAULT_TEXT_SIZE = 24
DEFAULT_NODE_SIZE = 10
DEFAULT_NODE_COLOR = 'black'
DEFAULT_EDGE_WIDTH = 0.5

class BaseGraphFig():
    '''
    The structure of any graph (be it G itself, or an ES tree, or a vertex cover cluster), is stored as a NetworkX graph.
    This is due to the robust primitives provided there.
    The visualization, on the other hand, is done in a Plotly Dash Figure, 
    which renders as a live webapp, using typical server-client (callbacks etc) patterns.
    This class is an interface combining the two, as well as the algorithm being implemented (e.g. ES trees).

    The class is meant to be subclassed, not instantiated directly.

    __init__: Performs input validation.
    '''
    def __init__(self,
                 nx_graph:nx.Graph, 
                 dynamic:bool,
                 ensure_update_valid:bool|None,
                 visualize: bool,
                 vis_per_step:bool,
                 dash_app: Dash|None,
                 dash_fig: go.Figure|None,
                 dash_fig_id:str|None,
                 dash_callback_caller:str|None,
                 dash_caller_inputvar:str|None,
                 ): 
        '''
        nx_graph: The graph. It is expected to be populated with the initial state __before__ you add it here.
        dynamic: If True, the graph is expected to be updated by an algorithm. If False, the graph is static.
        ensure_update_valid: If True, will check if the updates are valid in terms of graph structure (does the edge exist? ). If False, will not check.
        visualize: If True, will interact with the Dash instance and perform Figure updates as needed.
        vis_per_step: If True, will perform and wait on each step of the algorithm, (i.e. at a higher resolution than per update call)
        dash_app: The Dash instance.
        dash_fig: The Figure instance for the visualisation of this plot 
        dash_fig_id: The ID for the Figure instance in the Dash layout 
        dash_callback_caller: The ID of the Dash component that triggers a call of the callback function
        dash_caller_inputvar: The input (say, a count of the # of times a button is pressed) from the caller of the callback.

        '''
        if(type(self) is BaseGraphFig):
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")

        assert isinstance(nx_graph,nx.Graph)
        assert isinstance(visualize, bool)

        self.nx_graph = nx_graph
        for edge in self.nx_graph.edges:
            assert edge[0] != edge[1] # No self-loops allowed

        self.visualize = visualize 
        self.dynamic = dynamic
        self.vis_per_step = vis_per_step
        if self.vis_per_step:
            if not self.dynamic or not self.visualize:
                raise ValueError(f"{self.__class__} was instantiated with self.vis_per_step == {self.vis_per_step}, but self.dynamic == {self.dynamic} (expected True) and self.visualize == {self.visualize} (expected True). Cannot visualize per step.")

        if self.dynamic:
            assert isinstance(ensure_update_valid, bool)
            self.op_q = queue.Queue()
            self.shadow_graph_nodes = set(nx_graph.nodes)
            self.shadow_graph_edges = set() 
            for edge in self.nx_graph.edges:
                self.shadow_graph_edges.add( ( min(edge), max(edge) ) )

        if self.visualize:
            assert isinstance(dash_app, Dash)
            assert isinstance(dash_fig, go.Figure)
            self.dash_app: Dash = dash_app
            self.fig: go.Figure = dash_fig
            if self.dynamic: 
                assert isinstance(vis_per_step, bool)
                assert isinstance(dash_fig_id, str)
                assert isinstance(dash_callback_caller, str)
                assert isinstance(dash_caller_inputvar, str)
                
                self.vis_per_step = vis_per_step
                self.fig_id: str = dash_fig_id
                self.caller_id: str = dash_callback_caller 
                self.caller_input:str  = dash_caller_inputvar
                if(self.vis_per_step):
                    self.algo_nx_update_generator = None 
        
        if self.visualize:
            self.init_vis()

    def init_vis(self):
        self.init_layout()
        self.init_visdicts()
        self.init_traces()

    def init_traces(self):
        '''
        Draw the initial graph, if visualize is True 
        Needs to add traces to self.dash_fig.

        You need to recover traces from self.fig AFTER you add them.
        self.fig does a DEEP copy of the traces you initially pass.
        '''
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        
        if not self.visualize:
            raise ValueError(f"{self.__class__} was instantiated with self.visualize == {self.visualize}. Cannot visualize.")

        G = self.nx_graph
        for node_idx in G.nodes:
            node_data = G.nodes[node_idx]
            
            node_x, node_y = node_data['pos']            
            
            self.fig.add_trace(go.Scatter(
                x=[node_x,], y=[node_y,],
                mode='markers+text',
                text=node_data['name'],
                textfont=dict(size=node_data["textsize"]),
                name=node_data["name"],
                textposition='top right',
                customdata=[node_data["hovertext"],],
                hoverinfo=node_data["hoverinfo"],
                hovertemplate='%{customdata}' if node_data["hoverinfo"] != "none" else None, 
                marker=go.scatter.Marker(color=node_data["color"],size=node_data["size"])
            ))

            new_trace = self.fig.data[-1]
            assert new_trace['name'] == node_data['name']
            node_data["trace"] = new_trace

        for edge in G.edges:
            
            edge_data = G.edges[edge]
            
            u,v = edge 
            u_pos = G.nodes[u]['pos']
            v_pos = G.nodes[v]['pos']
            x = (u_pos[0],v_pos[0])
            y = (u_pos[1],v_pos[1])
            
            self.fig.add_trace(go.Scatter(
                            x=x, y=y,
                            name=edge_data["name"],
                            line=dict(width=edge_data["width"], color=edge_data["color"]), 
                            mode='lines',
                            hoverinfo=edge_data["hoverinfo"],
                            ))
            
            new_trace = self.fig.data[-1]
            assert new_trace['name'] == edge_data["name"]
            edge_data["trace"] = new_trace

    def init_layout(self):
        '''
        Set the layout of the figure. 
        '''
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        
        if not self.visualize:
            raise ValueError(f"{self.__class__} was instantiated with self.visualize == {self.visualize}. Cannot visualize.")
        
        G = self.nx_graph
        nx_layout = nx.spring_layout(G)
        for v in G:
            G.nodes[v]['pos'] = nx_layout[v].tolist()        

    def init_visdicts(self):
        
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        
        G = self.nx_graph
        for node_idx in G.nodes:
            node_data = G.nodes[node_idx]
            node_data["name"] = str(node_idx)
            node_data["hovertext"] = None 
            node_data["hoverinfo"] = "none"
            node_data["color"] = DEFAULT_NODE_COLOR
            node_data["size"] = DEFAULT_NODE_SIZE
            node_data["textsize"] = DEFAULT_TEXT_SIZE
        
        for edge in G.edges:
            u,v = edge 
            edge_data = G.edges[edge]
            edge_data["name"] = f"{u}_{v}"
            edge_data["color"] = 'black'
            edge_data["width"] = DEFAULT_EDGE_WIDTH
            edge_data["hoverinfo"] = "none"
    
    def update_overall(self):
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        if not self.dynamic:
            raise ValueError(f"{self.__class__} was instantiated with self.dynamic == {self.dynamic}. Cannot update.")
        
        if self.algo_nx_update_generator is None:
            self.algo_nx_update_generator = self.update_nx_by_algo()
            op, done, mutated_nodes_verts = next(self.algo_nx_update_generator)
            if(done):
                self.algo_nx_update_generator = self.update_nx_by_algo()
        if not self.visualize:
            return op, done, None
        else:
            patch:Patch = self.redraw_traces(mutated_nodes_verts)
            return op, done, patch

    def redraw_traces(self,graphobjs_to_redraw:list[int|tuple[int,int]])->Patch:
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        if not self.dynamic:
            raise ValueError(f"{self.__class__} was instantiated with self.dynamic == {self.dynamic}. Cannot update.")

    def update_nx_by_algo(self)->Generator[ tuple[ tuple[bool,int,int], bool, list[int|tuple[int,int]]],None,None]:
        '''
        If not self.visualize:
            Pop an op from self.op_q 
            Apply that op on the graph stored in self.nx_graph (using the algo as relevant)
            Return None 
        
        If self.visualize and not self.vis_per_step:
            Pop an op from self.op_q 
            Apply that op on the graph stored in self.nx_graph (using the algo as relevant)
            Apply all the changes to self.fig
            Return self.fig
        
        If self.visualize and self.vis_per_step:
            Pop an op from self.op_q 
            Considering the algo to be a sequence of atomic steps,
                Perform an atomic step onto self.nx_graph
                Perform a corresponding update to self.fig
                YIELD self.fig          <--- this will restart execution from _right here_. 
                                        You will have to use next(...) on the generator etc etc.
        '''
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        if not self.dynamic:
            raise ValueError(f"{self.__class__} was instantiated with self.dynamic == {self.dynamic}. Cannot update.")
        
    def add_op_to_queue(self, is_add:bool, u:int, v:int):
        
        if type(self) is BaseGraphFig:
            raise TypeError("GraphFig is meant to be subclassed, not instantiated directly.")
        if not self.dynamic:
            raise ValueError(f"{self.__class__} was instantiated with self.dynamic == {self.dynamic}. Cannot update.")
        
        if not isinstance(is_add, bool):
            raise TypeError(f"is_add must be a bool, got {is_add}")
        if not isinstance(u, int):
            raise TypeError(f"u must be an int, got {u}")
        if not isinstance(v, int):
            raise TypeError(f"v must be an int, got {v}")
        
        if u == v:
            raise ValueError(f"Self-loops not allowed. u == {u}, v == {v}")
        edge = (min(u,v), max(u,v))
        if(is_add and edge in self.shadow_graph_edges):
            raise ValueError(f"Edge {edge} already exists, cannot add again.")
        elif(not is_add and edge not in self.shadow_graph_edges):
            raise ValueError(f"Edge {edge} does not exist, cannot delete.")
        self.op_q.put((is_add, u, v))

class StaticGraphFig(BaseGraphFig):
    '''
    A static graph with integer vertex labels and no edge weights. No special coloring, no hoverinfo.
    '''
    def __init__(self, nx_graph: nx.Graph, visualize: bool, vis_per_step: bool, dash_app: Dash | None, dash_fig: go.Figure | None, dash_fig_id: str | None, dash_callback_caller: str | None, dash_caller_inputvar: str | None):
        super().__init__(nx_graph, False, False, visualize, vis_per_step, dash_app, dash_fig, dash_fig_id, dash_callback_caller, dash_caller_inputvar)

if __name__ == "__main__":
    G = nx.random_geometric_graph(10, 1)
    
    fig_id = 'fig'

    fig = go.Figure(  
                data=None,
                layout=go.Layout(
                    title=dict(
                        text=fig_id,
                        font=dict(
                            size=DEFAULT_TEXT_SIZE,
                        )
                    ),
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    dash_app = Dash(__name__)
    dash_app.layout = html.Div([dcc.Graph(id=fig_id, figure=fig)])
    fig = StaticGraphFig(G, True, False, dash_app, fig, fig_id, None, None)
    dash_app.run_server(debug=True)