# TODO: Test a queueless version of this. 

EDGE_WIDTH = 0.5
NODE_SIZE = 10
TEXT_SIZE = 22
FIG_OBJ_ID = 'fig'
INTERVAL_OBJ_ID = 'interval'
POLL_TIME_MS = 100
import plotly.graph_objects as go 

import threading
import queue
import random

from dash import Dash, html, dcc, Input, Output, callback, Patch
import time
import networkx as nx



def get_example_graph()->nx.Graph:
    G = nx.random_geometric_graph(10, 1)
    nx_layout = nx.spring_layout(G)
    for v in G.nodes:
        G.nodes.data()[v]['pos'] = nx_layout[v].tolist()
    return G
def example_nx_graph_to_graphdicts(G:nx.Graph)->tuple[dict,dict]:
    nodes_vis_data = {}
    edges_vis_data = {}

    nx_nodes = G.nodes 
    nx_nodes_data = G.nodes.data()

    nx_edges = G.edges 
    nx_edges_data = G.edges.data()

    colors = ['red','green','blue']
    rrange = (0,len(colors)-1)
    random.randint(*rrange)
    for i in nx_nodes:
        curr_nx_node_data = nx_nodes_data[i]
        stri = str(i)
        curr_node_vis_data = {
            'node_id':stri,
            'name':stri,
            'x': curr_nx_node_data['pos'][0],
            'y': curr_nx_node_data['pos'][1],
            'color':'black',
            'hoverinfo':None,
            'hovertext':stri + '_' + str(random.randint(100,110)),
        }
        nodes_vis_data[stri]=curr_node_vis_data

    for j in nx_edges:
        u, v = min(j), max(j)
        assert u != v 
        u_dt, v_dt = nx_nodes_data[u],nx_nodes_data[v]
        edge_id = f'{u}_{v}'
        u_x, u_y = u_dt['pos']
        v_x, v_y = v_dt['pos']
        curr_edge_vis_data = {
            "edge_id":edge_id,
            "name":edge_id,
            'x':(u_x,v_x),
            'y':(u_y,v_y),
            'color':'blue',
        }
        edges_vis_data[edge_id] = curr_edge_vis_data
    return (nodes_vis_data,edges_vis_data)

def new_fig_from_graphdicts(nodes_vis_data:dict,edges_vis_data:dict,figname:str|None=None)->tuple[go.Figure,dict[str:go.Trace],dict[str:go.Trace]]:
    '''
    G:nx.Graph -- must have called a layout on this already.
    '''
    
    fig = go.Figure(  
                data=None,
                layout=go.Layout(
                    title=dict(
                        text=figname,
                        font=dict(
                            size=TEXT_SIZE,
                        )
                    ),
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    node_traces_dict = {}
    for node_id in nodes_vis_data:
        node_dt = nodes_vis_data[node_id]    
        node_trace = go.Scatter(
            x=[node_dt['x'],], y=[node_dt['y'],],
            mode='markers+text',
            text=node_dt['name'],
            textfont=dict(size=TEXT_SIZE),
            name=node_dt['node_id'],
            textposition='top right',
            customdata=[node_dt["hovertext"],],
            hoverinfo=node_dt["hoverinfo"],
            hovertemplate='%{customdata}', 
            marker=go.scatter.Marker(color=node_dt["color"],size=NODE_SIZE)
        )
        fig= fig.add_trace(node_trace)
        figged_node_trace = fig.data[-1]
        assert(figged_node_trace['name'] == node_dt['node_id'])
        node_traces_dict[node_dt['node_id']] = figged_node_trace

    edge_traces_dict = {}
    for edge_id in edges_vis_data:
        edge_dt = edges_vis_data[edge_id]
        
        edge_trace = go.Scatter(
            x=edge_dt['x'], y=edge_dt['y'],
            name=edge_dt['name'],
            line=dict(width=EDGE_WIDTH, color=edge_dt['color']), 
            mode='lines',
            hoverinfo='none',
        )
        fig=fig.add_trace(edge_trace)
        figged_edge_trace = fig.data[-1]
        assert(figged_edge_trace['name'] == edge_dt['edge_id'])
        edge_traces_dict[edge_id] = edge_trace
    return (fig, node_traces_dict, edge_traces_dict)

def child(G:nx.Graph, Q:queue.Queue):

    @callback(
        Output(FIG_OBJ_ID,"figure"),
        Input(INTERVAL_OBJ_ID,"n_intervals"),
        prevent_initial_call=True,
        )
    def update_figure(interval):
        patch = Patch()
        if(not Q.empty()):
            task_lst = Q.get()
            for task_type, task_dict in task_lst:
                if(task_type == "NODE_UPDATE"):
                    print("HELO")
                    new_node_dict = task_dict
                    node_uid = new_node_dict['node_id']
                    node_trace = node_traces_dict[node_uid]
                    if('x' in new_node_dict and 'y' in new_node_dict):
                        node_trace['x'],node_trace['y'] = [new_node_dict['x'],], [new_node_dict['y'],]
                    if('color' in new_node_dict):
                        node_trace['marker']['color'] = new_node_dict['color']
                        # node_trace['marker'] = go.scatter.Marker(color=new_node_dict["color"],size=NODE_SIZE)
                    if('hoverinfo' in new_node_dict):
                        node_trace['hoverinfo'] = new_node_dict['hoverinfo']
                    if('hovertext' in new_node_dict):
                        node_trace['customdata']=[new_node_dict['hovertext'],]
                else:
                    raise NotImplementedError
            return fig
        return patch             

            

    nodes_vis_data, edges_vis_data = example_nx_graph_to_graphdicts(G)
    fig, node_traces_dict, edge_traces_dict = new_fig_from_graphdicts(nodes_vis_data,edges_vis_data,'EX_FIG')

    app = Dash(__name__)    
    app.layout = html.Div([
        dcc.Graph(figure=fig,id=FIG_OBJ_ID),
        dcc.Interval(INTERVAL_OBJ_ID,POLL_TIME_MS)
    ])
    app.run(debug=True,use_reloader=False)
    # tr = node_traces_dict['4']
    # tr["marker"]=go.scatter.Marker(color='red',size=NODE_SIZE)
if __name__ == '__main__':
    '''
    
    So the idea is that everything to do with networkx and the internal repr should happen in the main processs...
    ... while maintaining and updating the visualization is the job of the child process.

    Graphdicts would be maintained at the child.

    That being said, _whenever_ an edge is added/deleted/modified, or a node is added/deleted/modified, the corresponding
    changes will have to be pushed into the job queue.

    Better to make a wrapper around a graph G, with a subclass for vis.
    '''
    
    G = get_example_graph()
    
    Q = queue.Queue()
    childproc = threading.Thread(target=child,args=[G,Q])
    childproc.start()
    # child(G,Q)
    time.sleep(10)
    Q.put([("NODE_UPDATE", {'node_id':str(4),'color':'red','x':0.0,'y':0.0}),])
    print("tis in the Queue.")

    
    
