import itertools
import networkx as nx

def cut_value(g: nx.Graph, bits) -> int:
    return sum(1 for u, v in g.edges if bits[u] != bits[v])

def brute_force_maxcut(g: nx.Graph):
    n = g.number_of_nodes()
    best, best_bits = -1, None
    for bits in itertools.product([0, 1], repeat=n):
        val = cut_value(g, bits)
        if val > best:
            best, best_bits = val, bits
    return best, best_bits

def greedy_maxcut(g: nx.Graph):
    side = {}
    for node in g.nodes:
        a = sum(1 for nb in g[node] if side.get(nb) == 0)
        b = sum(1 for nb in g[node] if side.get(nb) == 1)
        side[node] = 1 if a >= b else 0
    bits = [side[i] for i in range(g.number_of_nodes())]
    return cut_value(g, bits), bits