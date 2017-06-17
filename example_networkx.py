import matplotlib.pyplot as plt
import networkx as nx
graph = {
    '1': ['2', '3', '4'],
    '2': ['5','11','12','13','14','15'],
    '3' : ['6','7','66','77'],
    '5': ['6', '8','66','77'],
    '4': ['7','66','77'],
    '7': ['9', '10']
    }

MG = nx.DiGraph(graph)


plt.figure(figsize=(8,8))
# pos=nx.graphviz_layout(MG,prog="twopi",root='1')

nodes = MG.nodes()
degree = MG.degree()
color = [degree[n] for n in nodes]
size = [2000 / (degree[n]+1.0) for n in nodes]
# nx.draw(MG, pos, nodelist=nodes, node_color=color, node_size=size,
#         with_labels=True, cmap=plt.cm.Blues, arrows=False)
nx.draw(MG, nodelist=nodes, node_color=color, node_size=size,
        with_labels=True, cmap=plt.cm.Blues, arrows=False)
plt.show()
