import pennylane as qml
from pennylane import numpy as pnp
import networkx as nx

def build_qaoa(g: nx.Graph, p: int):
    n = g.number_of_nodes()
    dev = qml.device("default.qubit", wires=n)
    edges = list(g.edges)

    # Cost Hamiltonian for Max-Cut: sum over edges of 0.5 * (1 - Z_u Z_v)
    # We minimise  sum Z_u Z_v / 2  (equivalent up to a constant)
    cost_h = qml.Hamiltonian(
        [0.5] * len(edges),
        [qml.PauliZ(u) @ qml.PauliZ(v) for u, v in edges],
    )

    def circuit(params):
        gammas, betas = params[:p], params[p:]
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p):
            for u, v in edges:
                qml.IsingZZ(2 * gammas[layer], wires=[u, v])  # cost layer
            for w in range(n):
                qml.RX(2 * betas[layer], wires=w)             # mixer layer

    @qml.qnode(dev)
    def expval_cost(params):
        circuit(params)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def probs(params):
        circuit(params)
        return qml.probs(wires=range(n))

    return expval_cost, probs, len(edges)

def expected_cut(expval: float, num_edges: int) -> float:
    # <cut> = num_edges/2 - <sum ZZ>/2
    return num_edges / 2 - expval