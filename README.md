# pygrank-downstream
This repository provides artificial intelligence solutions 
in large graphs by providing downstream applications 
of graph signal processing and node ranking. 
All operations are supported by the
[pygrank](https://github.com/MKLab-ITI/pygrank) library.

**License:** Apache Software License
<br>**Author:** Emmanouil (Manios) Krasanakis
<br>**Dependencies:** `pygrank`
<br><small>*Externally install optional depenendencies to support respective backends.*</small>


# :zap: Quickstart
```python
import pygrank as pg
from recipes import overlapping_community_detection

_, graph, group = next(pg.load_datasets_one_community(["citeseer"]))
train, test = pg.split(group, 0.1)

found = overlapping_community_detection(graph, train)

print("Recommended community:", found)
```
