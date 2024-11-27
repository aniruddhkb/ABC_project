from dash import Dash, html, dcc, Input, Output, callback

import plotly.graph_objects as go 
import time

import networkx as nx
from tqdm import tqdm 

gnodes_len = 0

G = nx.random_geometric_graph(10, 1)
nx_layout = nx.spring_layout(G)
for v in G.nodes:
    G.nodes.data()[v]['pos'] = nx_layout[v].tolist()

edge_x = []
edge_y = []
edge_colors = []
i = 0
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    if(i == 1):
        edge_colors.append("red")
    else:
        edge_colors.append("blue") 
    i = (i+1)%2

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color=edge_colors),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_label_str = []
node_hover_str = ['a','b','c','d','e']
node_color = ['red','green','yellow','black','blue']
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    node_label_str.append(str(node))

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_label_str,
    textposition='top right',
    customdata=node_hover_str,
    hovertemplate='%{customdata}', 
    marker=go.scatter.Marker(color=node_color,size=10)
)

fig = go.Figure(data=[node_trace,edge_trace],
             layout=go.Layout(
                title=dict(
                    font=dict(
                        size=5
                    )
                ),
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
    
