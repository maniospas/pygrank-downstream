import pygrank as pg


def community_detection(graph, known_members_set):
    ranks_set = [pg.ParameterTuner()(graph, known_members) for known_members in known_members_set]
    options = list(range(len(ranks_set)))

    def classify(params=[1]*len(ranks_set)):
        found_set = [list() for _ in known_members_set]
        for v in graph:
            found_set[max(options, key=lambda i:ranks_set[i][v]*params[i])].append(v)
        return found_set

    #def assess(found_set):
    #    return sum(pg.Conductance(graph)(found) for found in found_set)
    #params = pg.optimize(max_vals=[1]*len(ranks_set), loss=lambda params: assess(classify(params)))
    return classify()


def overlapping_community_detection(graph, known_members, top=None):
    if len(known_members) < 50:  # threshold by AutoGF paper
        graph_filter = pg.PageRank(0.9)
    else:
        graph_filter = pg.ParameterTuner().tune(graph, known_members)
    ranks = pg.Sweep(graph_filter)(graph, {v: 1 for v in known_members})
    ranks = pg.Normalize("range")(ranks)
    if top is not None:
        return sorted(list(graph), key=lambda v: -ranks[v])[:top]

    def loss(params):
        return pg.Conductance(graph)(pg.Threshold(params[0]).transform(ranks))
    params = pg.optimize(max_vals=[1], loss=loss)
    return [v for v in graph if ranks[v] >= params[0]]
