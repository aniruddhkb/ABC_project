# Visualization:

## Properties of an edge: 

Editable properties:
    x
    y
    color

Init Custom properties:
    name = n1_n2
    x = (x1, x2, None)
    y = (y1, y2, None)
    color = '#[hexadec]'


Example edge trace in the beginning:

edge_trace = go.Scatter(
    name=n1_n2
    x=x, y=y,
    line=dict(width=EDGE_WIDTH, color=color),
    hoverinfo='none',
    mode='lines')

## Properties of a node:

Editable properties:
    x
    y
    color
    hoverinfo
    hovertext

Init Custom properties:
    name=n
    x
    y
    color
    hoverinfo 'none'|None
    hovertext 

node_trace = go.Scatter(
    x=x, y=y,
    mode='markers+text',
    text=str(n),
    textposition='top right',
    customdata=node_hover_str,
    hovertemplate='%{customdata}', 
    marker=go.scatter.Marker(color=node_color,size=10)
)





