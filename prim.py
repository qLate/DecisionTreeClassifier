import networkx as nx
import heapq


def prim_minimum_spanning_tree(G: nx.Graph) -> nx.Graph:
    vertecies = list(G.nodes)
    added_vertecies = set([0])

    edges_heap = []
    for edge in G.edges(0, data="weight"):
        heapq.heappush(edges_heap, edge[::-1])

    min_tree = nx.Graph()
    min_tree.add_nodes_from(vertecies)
    for _ in range(0, len(vertecies)-1):
        min_edge = heapq.heappop(edges_heap)
        while min_edge[1] in added_vertecies:
            min_edge = heapq.heappop(edges_heap)

        added_vertecies.add(min_edge[1])
        end_vertex_edges = G.edges(min_edge[1], data="weight")
        for edge in end_vertex_edges:
            if edge[1] not in added_vertecies:
                heapq.heappush(edges_heap, edge[::-1])

        min_tree.add_edge(min_edge[1], min_edge[2], weight=min_edge[0])
    return min_tree
