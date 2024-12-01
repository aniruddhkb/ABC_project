import plotly.graph_objects as go 
import random

from dash import Dash, html, dcc, Input, Output, callback
import time

import networkx as nx
from tqdm import tqdm 

EDGE_WIDTH = 0.5
NODE_SIZE = 10
TEXT_SIZE = 22

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
        fig.add_trace(node_trace)
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
        fig.add_trace(edge_trace)
        figged_edge_trace = fig.data[-1]
        assert(figged_edge_trace['name'] == edge_dt['edge_id'])
        edge_traces_dict[edge_id] = edge_trace
    return (fig, node_traces_dict, edge_traces_dict)

if __name__ == '__main__':
    G = get_example_graph()
    nodes_vis_data, edges_vis_data = example_nx_graph_to_graphdicts(G)
    fig, node_traces_dict, edge_traces_dict = new_fig_from_graphdicts(nodes_vis_data,edges_vis_data,'EX_FIG')
    tr = node_traces_dict['4']
    tr["marker"]=go.scatter.Marker(color='red',size=NODE_SIZE)
    print(tr)
    print(fig.data[4])
    fig.show()
