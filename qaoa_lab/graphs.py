import networkx as nx

def random_graph(n: int, edge_prob: float = 0.5, seed: int = 0) -> nx.Graph:
    g = nx.erdos_renyi_graph(n, edge_prob, seed=seed)
    # Ensure at least one edge so Max-Cut is non-trivial
    while g.number_of_edges() == 0:
        seed += 1
        g = nx.erdos_renyi_graph(n, edge_prob, seed=seed)
    return g