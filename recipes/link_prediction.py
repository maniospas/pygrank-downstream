import pygrank as pg


def link_prediction(graph, node, top=5):
    algorithm = pg.SeedOversampling(pg.HeatKernel(15), "neighbors")
    ranks = algorithm(graph, {node: 1})
    return sorted(graph, key=lambda v: -ranks[v] if not graph.has_edge(node, v) else 0)[:top]
