from matplotlib import pyplot as plt
import networkx as nx

from random_graph_generator import gnp_random_connected_graph


def kruskal_minimum_spanning_tree(G):

    edges = sorted(list(G.edges()),
                   key=lambda x: G.get_edge_data(x[0], x[1])['weight'])
    vertexes = list({i} for i in list(G.nodes()))
    karkas = []
    length = len(vertexes)
    for edge in edges:
        if len(karkas) == length-1:
            break
        point = True
        for index in range(len(vertexes)):
            if edge[0] in vertexes[index] and edge[1] in vertexes[index]:
                break
            if (edge[0] in vertexes[index] or edge[1] in vertexes[index]) and point:
                first_index = index
                point = False
                continue
            if (edge[0] in vertexes[index]) or (edge[1] in vertexes[index]):
                vertexes[first_index].update(vertexes[index])
                vertexes.pop(index)
                karkas.append(edge)
                break
    vertecies = list(G.nodes)
    min_tree = nx.Graph()
    min_tree.add_nodes_from(range(len(vertecies)))
    min_tree.add_edges_from(karkas)
    return min_tree


# input = gnp_random_connected_graph(100, 0)
# mstp = kraskal_algorithm(input)
# print(sum([edge[2] for edge in input.edges(data="weight")
#       if (edge[0], edge[1]) in mstp.edges()]))
# nx.draw(mstp, node_color='lightblue',
#         with_labels=True,
#         node_size=500)
# plt.savefig("prim.png")
# plt.clf()
# mstp = nx.minimum_spanning_tree(input, algorithm="prim")
# print(sum([edge[2] for edge in input.edges(data="weight")
#       if (edge[0], edge[1]) in mstp.edges()]))

# nx.draw(mstp, node_color='lightblue',
#         with_labels=True,
#         node_size=500)

# plt.savefig("prim1.png")
