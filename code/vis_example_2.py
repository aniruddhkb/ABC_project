from dash import Dash, html, dcc, Input, Output, callback

import plotly.graph_objects as go 
import time

import networkx as nx
from tqdm import tqdm 

G = nx.random_geometric_graph(10, 0.9)
nx_layout = nx.spring_layout(G)

for i in G.nodes:
    G.nodes[i].data
