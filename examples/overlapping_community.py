import pygrank as pg
from recipes import overlapping_community_detection

_, graph, group = next(pg.load_datasets_one_community(["amazon"]))
print(len(group))
train, test = pg.split(group, 0.1)
found = overlapping_community_detection(graph, train)
train, test = set(train), set(test)
new_nodes = [v for v in found if v not in train]
TP = len([v for v in new_nodes if v in test])
print("Precision", TP/len(new_nodes))
print("Recall", TP/len(test))
print("Match size", len(found)/len(group))

#print(group)
#print(new_nodes)