from dash import Dash, html, dcc, Input, Output, Patch, callback
import plotly.graph_objects as go 
import networkx as nx

def get_figure():    
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
    return fig 

app = Dash(__name__)    
fig = get_figure()
RAE_OBJ_ID = 'rae' 
FIG_OBJ_ID = 'fig'
INTERVAL_ID = 'interval'
app.layout = html.Div([
    html.Button('Remove an Edge.',id=RAE_OBJ_ID),
    dcc.Graph(figure=fig,id=FIG_OBJ_ID),
    dcc.Interval(INTERVAL_ID,5000)
])

@callback(
        Output(FIG_OBJ_ID,"figure"),
        Input(INTERVAL_ID,"n_intervals"),
        )
def update_figure(interval):
    sct = fig.data[1]
    print(len(sct.x))
    sct.x, sct.y = sct.x[3:], sct.y[3:]
    patched_figure = Patch()
    patched_figure["data"][1] = sct
    return patched_figure

if __name__ == "__main__":
    app.run(debug=True)