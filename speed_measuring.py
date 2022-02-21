from prim import prim_minimum_spanning_tree
from kruskal import kruskal_minimum_spanning_tree

from random_graph_generator import gnp_random_connected_graph
import time
import matplotlib.pyplot as plt


def get_avarage_time(edges: int, algorithm: str, iterations: int):
    time_taken = 0
    for i in range(iterations):
        G = gnp_random_connected_graph(edges, 1, False)

        start = time.time()
        G = prim_minimum_spanning_tree(
            G) if algorithm == "prim" else kruskal_minimum_spanning_tree(G)
        end = time.time()

        time_taken += end - start

    return time_taken/iterations


plt.xlabel('vertecies')
plt.ylabel('time')
plt.title('prim(red) VS kruskal(blue)')

step = 1
x = []
y_prim = []
y_kruskal = []
for vertecies in range(step, 500, step):
    x.append(vertecies)
    y_prim.append(get_avarage_time(vertecies, "prim", 10))
    y_kruskal.append(get_avarage_time(vertecies, "kraskal", 10))

plt.plot(x, y_prim, label="prim", color="red")
plt.plot(x, y_kruskal, label="kruskal", color="blue")

plt.savefig("graph")
