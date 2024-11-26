from dash import Dash, html, dcc, Input, Output, callback

import plotly.graph_objects as go 
import time
# import plotly.io as pio
# pio.renderers.default = 'jupyterlab'

import networkx as nx
from tqdm import tqdm 

# G = nx.Graph()
# G.add_node(1)
# G.add_node(2)
# G.add_node(3)

# G.add_edge(1,2)
# G.add_edge(2,3)

gnodes_len = 0

while(gnodes_len < 10): 
    nG = nx.random_geometric_graph(15, 0.2)
    G:nx.Graph = nx.subgraph(nG,nx.node_connected_component(nG,1))
    gnodes_len = len(G.nodes)

bfs_layout = nx.bfs_layout(G,1,align='horizontal',scale=-0.5)
for v in G.nodes:
    G.nodes.data()[v]['pos'] = bfs_layout[v].tolist()

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_label_str = []
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
    hoverinfo='none', 
    marker=go.scatter.Marker(color='black',size=3)
)
fig = go.FigureWidget(data=[node_trace,edge_trace],
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
while tqdm(len(fig.data[1].x) > 0):
    print(len(fig.data[1].x))
    sct = fig.data[1]
    newx, newy = sct.x[3:], sct.y[3:]
    time.sleep(5)
    fig.update_traces(patch={"x":newx,"y":newy},selector=1,overwrite=True)
    fig.show()
    
