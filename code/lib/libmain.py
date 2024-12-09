from graphfig import *
from bfs import *
from even_shiloach import *
from recursive_spanner import *
from decr_const import *
import networkx as nx
'''
Parameters for each:



'''

class DynAPSPAlgo(DynAlgo):
    def __init__(self, base_graph:nx.Graph)