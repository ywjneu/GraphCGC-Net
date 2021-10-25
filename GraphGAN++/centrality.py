from methods import *


def topological_measures(data, threshold):
    # ROI is the number of brain regions (i.e.,35 in our case)
    ROI = 8
    n_CC = np.empty((0, ROI), int)
    n_BC = np.empty((0, ROI), int)
    n_DC = np.empty((0, ROI), int)
    p_CC = np.empty((0, ROI), int)
    p_BC = np.empty((0, ROI), int)
    p_DC = np.empty((0, ROI), int)
    n_topology = []
    p_topology = []
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            if j == 0:
                A = data[i][j].cpu()
                A = torch.tensor(A, dtype=torch.float64)
                A = torch.where(A < threshold, 0., A)  # 对于SuperGrape而言,较小的连接性可以忽略。
                A = A.cpu().detach().numpy()
                np.fill_diagonal(A, 0)

                # create a graph from similarity matrix
                G = nx.from_numpy_matrix(A)
                U = G.to_undirected()

                # Centrality #
                # compute closeness centrality and transform the output to vector
                cc = nx.closeness_centrality(U)
                closeness_centrality = np.array([cc[g] for g in U])
                # compute betweeness centrality and transform the output to vector
                bc = nx.betweenness_centrality(U)
                betweenness_centrality = np.array([bc[g] for g in U])

                dc = nx.degree_centrality(U)
                degree_centrality = np.array([dc[g] for g in U])

                # create a matrix of all subjects centralities
                n_CC = np.vstack((n_CC, closeness_centrality))
                n_BC = np.vstack((n_BC, betweenness_centrality))
                n_DC = np.vstack((n_DC, degree_centrality))
            elif j == 1:
                A = data[i][j]
                A = A.cpu().detach().numpy()
                np.fill_diagonal(A, 0)

                # create a graph from similarity matrix
                G = nx.from_numpy_matrix(A)
                U = G.to_undirected()

                # Centrality #
                # compute closeness centrality and transform the output to vector
                cc = nx.closeness_centrality(U)
                closeness_centrality = np.array([cc[g] for g in U])
                # compute betweeness centrality and transform the output to vector
                bc = nx.betweenness_centrality(U)
                betweenness_centrality = np.array([bc[g] for g in U])
                # compute egeinvector centrality and transform the output to vector
                dc = nx.degree_centrality(U)
                degree_centrality = np.array([dc[g] for g in U])

                # create a matrix of all subjects centralities
                p_CC = np.vstack((p_CC, closeness_centrality))
                p_BC = np.vstack((p_BC, betweenness_centrality))
                p_DC = np.vstack((p_DC, degree_centrality))

    n_topology.append(n_CC)  # 0
    n_topology.append(n_BC)  # 1
    n_topology.append(n_DC)  # 2
    p_topology.append(p_CC)  # 0
    p_topology.append(p_BC)  # 1
    p_topology.append(p_DC)  # 2
    return n_topology, p_topology

