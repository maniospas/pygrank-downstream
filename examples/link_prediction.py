import pygrank as pg
import networkx as nx
from recipes import link_prediction

graph = next(pg.load_datasets_graph(["citeseer"], graph_api=nx))
precisions = list()
recalls = list()
for node in graph:
    if graph.degree(node) < 10:
        continue
    _, test = pg.split(list(graph.neighbors(node)))
    test = set(test)
    for v in test:
        graph.remove_edge(node, v)
    recommendation = link_prediction(graph, node)
    TP = len([v for v in recommendation if v in test])
    precisions.append(TP/len(recommendation))
    recalls.append(TP/len(test))
    for v in test:
        graph.add_edge(node, v)
    print("Avg. precision", sum(precisions)/len(precisions))
    print("Avg. recall", sum(recalls)/len(recalls))
