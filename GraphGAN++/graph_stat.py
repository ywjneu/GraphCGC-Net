import igraph
import networkx as nx
import numpy as np
import powerlaw
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


# import matplotlib.pyplot as plt
# plt.switch_backend('agg')


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """
    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0).flatten()
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0).flatten()
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees + .0001) / (2 * float(m))))
    return H_er


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes  x
             * Size of the largest connected component (LCC) x
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    A = A_in.copy()

    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}
    # statistics_degrees 计算邻接矩阵度的最大值，最小值，平均值
    d_max, d_min, d_mean = statistics_degrees(A)

    # node number & edger number
    statistics['node_num'] = A_graph.number_of_nodes()
    statistics['edge_num'] = A_graph.number_of_edges()

    # -----------------------
    # # 度的角度：
    # -----------------------

    # LCC
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]

    # TC
    statistics['TC'] = statistics_triangle_count(A)

    # PLA
    statistics['PLA'] = statistics_power_law_alpha(A)

    # GINI
    statistics['GINI'] = statistics_gini(A)

    # CPL
    statistics['CPL'] = statistics_compute_cpl(A)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['TC'] / ((statistics_claw_count(A)) + 1)

    #  -----------------------
    # 边的角度：
    # -----------------------
    # REDE \\\\\Relative edge distribution entropy
    statistics['REDE'] = statistics_edge_distribution_entropy(A)
    # 节点强度
    statistics['NS'] = d_mean

    # -----------------------
    # # 中心性的角度：
    # -----------------------
    # Centrality #
    # compute closeness centrality and transform the output to vector
    cc = nx.closeness_centrality(A_graph)
    closeness_centrality = np.array([cc[g] for g in A_graph]).mean()
    statistics['CC'] = closeness_centrality

    # compute betweeness centrality and transform the output to vector
    bc = nx.betweenness_centrality(A_graph)
    Ebc = nx.edge_betweenness_centrality(A_graph)
    betweenness_centrality = np.array([bc[g] for g in A_graph]).mean()
    statistics['BC'] = betweenness_centrality

    # compute egeinvector centrality and transform the output to vector
    # ec = nx.eigenvector_centrality(A_graph)
    # eigenvector_centrality = np.array([ec[g] for g in A_graph]).mean()
    # statistics['EC'] = eigenvector_centrality
    # compute degree centrality and transform the output to vector
    cc = nx.degree_centrality(A_graph)
    degree_centrality = np.array([cc[g] for g in A_graph]).mean()
    statistics['DC'] = degree_centrality

    return statistics
