import pygraphviz as pg
#import networkx as nx
import load_categories as lc


h = lc.load(True)
G = pg.AGraph(directed=True, strict=True)
for k in h.keys():
    #G.add_node(k, label='')
    for child in h[k]['children']:
        G.add_edge(k, child)
    G.add_edge(h[k]['parent'], k)
G.layout(prog='dot')
#G.graph_attr.update(size='2,2')
G.draw('ch.svg')