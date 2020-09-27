import networkx as nx
from scipy import sparse as sp
import numpy as np


def quotient_graph(G, partition):
    """Calculate the quotient graph according to a partition, ~100x faster than networkx
    for large graphs

    Args:
        G: nx.Graph
        partition: pd.Series with partition indices ordered same as G.nodes()

    Returns:
        Gb: nx.Graph

    """
    W = nx.to_scipy_sparse_matrix(G)
    B = sp.lil_matrix((len(partition), partition.nunique()))
    for i in partition.unique():
        B[np.where(partition == i)[0], i] = 1
    Wb = (B.T @ (W @ B))
    Gb = nx.from_scipy_sparse_matrix(Wb, create_using=G.__class__)

    return Gb
