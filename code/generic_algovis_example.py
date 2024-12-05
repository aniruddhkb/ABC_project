'''
Here, the graph update algorithm does nothing in particular, except for demonstrating the API for updates.
'''

import sys 
sys.path.append('./lib')

from lib import *
from dash import Dash, html, dcc, Input, Output
import networkx as nx

if __name__ == '__main__':
    
    app = Dash(__name__)
    nx_graph = nx.Graph()

    figs_dict = {
        'base':new_default_fig(),
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
