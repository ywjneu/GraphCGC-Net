import torch
import torch.nn.functional as F
from sklearn.manifold import SpectralEmbedding
import warnings

from torch.autograd import Variable

warnings.filterwarnings("ignore")
from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
import copy

from networkx.drawing.nx_agraph import graphviz_layout

from graph_stat import *


def compute(adj, base_adj):
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)  # copy.deepcopy 表示深复制，意思是 adj_ 与 adj 相互独立，一个发生改变不影响另一个

    adj_ -= np.diag(np.diag(adj_))  # 将 adj_ 对角线上的元素置零

    assert ((adj_ == adj_.T).all())

    d = compute_graph_statistics(adj_)  # 计算生成图的各属性值

    if not isinstance(base_adj, np.ndarray):
        base_adj_ = base_adj.data.cpu().numpy()
    else:
        base_adj_ = copy.deepcopy(base_adj)  # copy.deepcopy 表示深复制，意思是 adj_ 与 adj 相互独立，一个发生改变不影响另一个

    bd = compute_graph_statistics(base_adj_)  # 计算原始图的各属性值
    diff_d = {}
    for k in list(d.keys()):
        diff_d[k] = round(abs(d[k] - bd[k]), 4)  # 计算原始图与生成图各属性值的绝对值差异
    return diff_d, bd


def show_graph(adj, base_adj, remove_isolated=True, args=None, i=0):  # adj 是生成图的邻接矩阵，base_adj 是原始图的邻接矩阵
    # 绘制生成图的规则图
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)  # copy.deepcopy 表示深复制，意思是 adj_ 与 adj 相互独立，一个发生改变不影响另一个

    adj_ -= np.diag(np.diag(adj_))  # 将 adj_ 对角线上的元素置零

    gr = nx.from_numpy_array(adj_)  # 从数组 adj_ 取出元素生成图
    assert ((adj_ == adj_.T).all())

    if remove_isolated:
        gr.remove_nodes_from(list(nx.isolates(gr)))  # 去掉图 gr 中的孤立节点
    nx.draw(gr, node_size=10)  # nx.draw 用于绘制图，node_size 是指节点的尺寸
    plt.title('gen')  # 将上面绘制的生成图打印出来
    plt.savefig('./{}/{}_generated_graph'.format(args.output_dir, i))
    plt.close()

    # 绘制原始图的规则图
    if not isinstance(base_adj, np.ndarray):
        base_adj_ = base_adj.data.cpu().numpy()
    else:
        base_adj_ = copy.deepcopy(base_adj)  # copy.deepcopy 表示深复制，意思是 adj_ 与 adj 相互独立，一个发生改变不影响另一个

    base_gr = nx.from_numpy_array(base_adj_)
    nx.draw(base_gr, node_size=10)
    plt.title('origi')
    plt.savefig('./{}/{}_original_graph'.format(args.output_dir, i))
    plt.close()  # 将原始图打印出来


# base_gr1 = nx.from_numpy_array(base_adj_)            # 测试，目的是观察两次绘制的原始图的规则是否相同，经过测试，发现并不相同
# nx.draw(base_gr1, node_size=10)
# plt.title('base1')
# plt.show()

def show_graph1(adj, base_adj, remove_isolated=True, args=None, i=0):  # adj 是生成图的邻接矩阵，base_adj 是原始图的邻接矩阵
    # 绘制原始图的坐标图
    if not isinstance(base_adj, np.ndarray):
        base_adj_ = base_adj.data.cpu().numpy()
    else:
        base_adj_ = copy.deepcopy(base_adj)

    base_G = nx.from_numpy_matrix(base_adj_)
    # if remove_isolated:
    # 	base_G.remove_nodes_from(list(nx.isolates(base_G)))         # 去掉图 gr 中的孤立节点
    # base_pos = nx.spring_layout(base_G)
    # base_pos = nx.random_layout(base_G)
    # base_pos = nx.graphviz_layout(base_G, prog='dot')
    base_pos = graphviz_layout(base_G, prog='neato')
    # base_pos = nx.spectral_layout(base_G)
    # nx.draw_networkx_nodes(base_G, base_pos, node_size=10, edge_vmin=0.0, edge_vmax=0.1, node_color='', alpha=0.5)
    # nx.draw_networkx_edges(base_G, base_pos, alpha=0.1)
    # plt.title('origi')
    nx.draw(base_G, base_pos, node_size=8, width=0.2, edge_color='grey')
    plt.savefig('./{}/{}_original_graph'.format(args.output_dir, i), dpi=1000)
    plt.close()

    # 绘制原始图的坐标图
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)  # copy.deepcopy 表示深复制，意思是 adj_ 与 adj 相互独立，一个发生改变不影响另一个

    adj_ -= np.diag(np.diag(adj_))  # 将 adj_ 对角线上的元素置零

    assert ((adj_ == adj_.T).all())

    gen_G = nx.from_numpy_matrix(adj_)
    # if remove_isolated:
    # 	gen_G.remove_nodes_from(list(nx.isolates(gen_G)))   # 去掉gen_G中的孤立节点
    # gen_pos = nx.spring_layout(gen_G)
    # gen_pos = nx.random_layout(gen_G)
    # gen_pos = nx.graphviz_layout(gen_G, prog='dot')
    # gen_pos = graphviz_layout(gen_G, prog='neato')
    # nx.draw_networkx_nodes(gen_G, gen_pos, node_size=10, edge_vmin=0.0, edge_vmax=0.1, node_color='blue', alpha=0.5)
    # nx.draw_networkx_edges(gen_G, gen_pos, alpha=0.1)
    nx.draw(gen_G, base_pos, node_size=8, width=0.2, edge_color='grey')
    # plt.title('gen')                                            # 将上面绘制的生成图打印出来
    plt.savefig('./{}/{}_generated_graph'.format(args.output_dir, i), dpi=1000)
    plt.close()


import scipy.sparse as sp


def get_matrix_triad(coo_matrix, data=False):
    if not sp.isspmatrix_coo(coo_matrix):
        coo_matrix = sp.coo_matrix(coo_matrix)

    # nx3的矩阵  列分别为 矩阵行，矩阵列及对应的矩阵值
    temp = np.vstack((coo_matrix.row, coo_matrix.col, coo_matrix.data)).transpose()
    return temp.tolist()


def show_graph2(adj, base_adj, remove_isolated=True, args=None, i=0):  # adj 是生成图的邻接矩阵，base_adj 是原始图的邻接矩阵
    # 绘制原始图
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)  # copy.deepcopy 表示深复制，意思是 adj_ 与 adj 相互独立，一个发生改变不影响另一个

    adj_ -= np.diag(np.diag(adj_))  # 将 adj_ 对角线上的元素置零
    assert ((adj_ == adj_.T).all())

    edgs = get_matrix_triad(adj_)
    H = nx.path_graph(adj_.shape[0])

    gen_G = nx.Graph()
    gen_G.add_nodes_from(H)
    # gen_G.add_edges_from(edgs)

    colors = np.arange(adj_.shape[0])
    nx.draw(gen_G, pos=nx.spring_layout(gen_G), node_color=colors)
    plt.title('gen')  # 将上面绘制的生成图打印出来
    plt.savefig('./{}/{}_generated_graph'.format(args.output_dir, i), dpi=1000)
    plt.close()

    # 绘制生成图
    if not isinstance(base_adj, np.ndarray):
        base_adj_ = base_adj.data.cpu().numpy()
    else:
        base_adj_ = copy.deepcopy(base_adj)

    edgs = get_matrix_triad(base_adj_)
    print(edgs)
    H = nx.path_graph(base_adj_.shape[0])

    origi_G = nx.Graph()
    origi_G.add_nodes_from(H)
    # origi_G.add_edges_from(edgs)

    colors = np.arange(base_adj_.shape[0])
    nx.draw(origi_G, pos=nx.spring_layout(origi_G), node_color=colors)
    plt.title('origi')  # 将上面绘制的生成图打印出来
    plt.savefig('./{}/{}_original_graph'.format(args.output_dir, i), dpi=1000)
    plt.close()


def make_symmetric(m):
    m_ = torch.transpose(m)
    w = torch.max(m_, m_.T)
    return w


def make_adj(x, n):
    res = torch.zeros(n, n)
    i = 0
    for r in range(1, n):
        for c in range(r, n):
            res[r, c] = x[i]
            res[c, r] = res[r, c]
            i += 1
    return res


def cat_attr(x, attr_vec):
    # print("attr_vec in cat_attr:",attr_vec)    # 这是为了辅助找bug，当不使用条件向量时，刚开始遇到了维度不匹配的bug
    if attr_vec is None:
        return x
    attr_mat = attr_vec.repeat(x.size()[0], 1)
    x = torch.cat([x, attr_mat], dim=1)
    return x


def get_spectral_embedding(adj, d):  # 返回邻接矩阵的拉普拉斯映射 embedding 矩阵，shape是[N, D]
    """
	Given adj is N*N, return its feature mat N*D, D is fixed in gan_model
	:param adj:
	:return:
	"""
    temp_ = []
    for reduce in adj:
        adj_ = reduce.data.cpu().numpy()
        emb = SpectralEmbedding(
            n_components=d)  # sklearn.manifold.SpectralEmbedding 用于返回邻接矩阵的拉普拉斯映射，即 embedding矩阵，维度是 n_components=d
        res = emb.fit_transform(adj_)
        temp_.append(res)
    x = torch.from_numpy(np.array(temp_)).float().cuda()
    x = Variable(x)
    return x


"""
def normalize(adj):                                     # 邻接矩阵归一化处理，即 D^{-1/2}(A+I)D^{1/2}
	adj = adj.data.cpu().numpy()
	adj_ = adj + np.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
	degree_mat_sqrt = np.diag(np.power(rowsum, 0.5).flatten())
	adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_sqrt)
	return torch.from_numpy(adj_normalized).float().cuda()
	# return torch.from_numpy(adj_normalized).float().cpu()
"""


def normalize(adj):
    # 邻接矩阵归一化处理，即 D^{-1/2}(A+I)D^{1/2}
    one_batch = []
    for batchsize in adj:
        temp_norm = []
        for reduce in batchsize:
            reduce = reduce + torch.eye(reduce.shape[0]).cuda()
            rowsum = reduce.sum(1)
            degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
            degree_mat_sqrt = torch.diag(torch.pow(rowsum, 0.5).flatten())
            adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, reduce), degree_mat_sqrt)
            temp_norm.append(adj_normalized.unsqueeze(0))
        temp_norm_ = torch.cat(temp_norm, 0)
        one_batch.append(temp_norm_.unsqueeze(0))
    one_batch = torch.cat(one_batch, 0)
    return one_batch

def normalize_(adj):
    # 邻接矩阵归一化处理，即 D^{-1/2}(A+I)D^{1/2}
    adj = adj + torch.eye(adj.shape[0]).cuda()
    rowsum = adj.sum(1)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
    degree_mat_sqrt = torch.diag(torch.pow(rowsum, 0.5).flatten())
    adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, adj), degree_mat_sqrt)
    return adj_normalized
    return


def keep_topk_conns(adj, k=3):  # 只保留图的最大的3个子图，出去其余的子图(例如孤立节点)
    g = nx.from_numpy_array(adj)  # 用邻接矩阵构建图
    to_removes = [cp for cp in sorted(nx.connected_components(g), key=len)][:-k]  # nx.connected_components(g) 输出g的连通子图
    for cp in to_removes:
        g.remove_nodes_from(cp)
    adj = nx.to_numpy_array(g)
    return adj


def remove_small_conns(adj, keep_min_conn=4):
    g = nx.from_numpy_array(adj)
    for cp in list(nx.algorithms.components.connected_components(g)):
        if len(cp) < keep_min_conn:
            g.remove_nodes_from(cp)
    adj = nx.to_numpy_array(g)
    return adj


def top_n_indexes(arr, n):
    idx = np.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]

    # topk_adj 的作用是输出 0-1 二值矩阵，并且确保生成图的边数等于训练邻接矩阵的变数


def topk_adj(adj, k):  # k // 2 是当前的训练邻接矩阵的边数，topk_adj 函数的作用是选出 adj 中最大的 k // 2 个元素（注，此时解码器输出的adj并不是二值矩阵）并置1，其余元素置零
    adj_comb = []
    for temp in adj:
        adj_ = temp.data.cpu().numpy()
        adj_ = (adj_ - np.min(adj_)) / np.ptp(adj_)  # np.ptp(adj_) 计算 adj_ 中最大元素与最小元素的差，这一行的作用是将 adj_ 中的每个元素与最小元素的差除以 adj_ 中最大元素与最小元素的差

        tri_adj = np.triu(adj_)  # np.triu返回上三角矩阵
        inds = top_n_indexes(tri_adj, k // 2)  # inds 是 tri_adj 中最大的 k // 2 个元素的索引
        res = np.zeros(temp.shape)
        for ind in inds:
            i = ind[0]
            j = ind[1]
            res[i, j] = adj_[i, j]
            res[j, i] = adj_[j, i]
        adj_comb.append(res)
    return Variable(torch.from_numpy(np.array(adj_comb)).float(), requires_grad=False).cuda()


def binary_adj(adj):
    adj_ = adj.data.cpu().numpy()
    assert ((adj_ == adj_.T).all())
    adj_ -= np.diag(np.diag(adj_))  # adj_ 的对角线元素置零
    res = np.where(adj_ <= 0.75, 0., 1.)
    assert ((res == res.T).all())
    res = Variable(torch.from_numpy(res).float(), requires_grad=True).cuda()
    return res


def test_gen(model, n, attr_vec, z_size, twice_edge_num, bd=None):
    fixed_noise = torch.randn((n, z_size), requires_grad=True).cuda()
    # fixed_noise = torch.randn((n, z_size), requires_grad=True).cpu()
    if attr_vec is not None:
        fixed_noise = cat_attr(fixed_noise, attr_vec.cuda())
    # fixed_noise = cat_attr(fixed_noise, attr_vec)
    a_ = model.decoder(fixed_noise)
    # print(F.sigmoid(a_))
    a_ = topk_adj(F.sigmoid(a_), twice_edge_num)
    # print(a_)
    if bd:
        show_graph(a_, bd)
        print("运行test_gen")
    else:
        print("运行test_gen")
        show_graph(a_)


def gen_adj(model, n, e, z_size):  # gen_adj 的作用是生成具有 n 个节点，e 条边的邻接矩阵
    # gan_model 是生成器模型，n 是训练邻接矩阵的节点数，e 是训练邻接矩阵的边数，attr_vec 是条件向量，z_size 是噪声向量的维度
    fixed_noise = torch.randn((n, z_size), requires_grad=False).cuda()
    # print("fixed_noise.shape:",fixed_noise.shape)  # 这一行和下一行，174行是为了辅助找bug，当不使用条件向量时，刚开始遇到了维度不匹配的bug
    # print("attr_vec in gen_adj:", attr_vec)
    # fixed_noise = torch.randn((n, z_size), requires_grad=True).cpu()
    # fixed_noise = cat_attr(fixed_noise, attr_vec)   # 噪声向量拼接条件向量
    # print("fixed_noise.shape:",fixed_noise.shape)
    rec_adj = model(fixed_noise)  # 生成器(解码器)输出邻接矩阵，此时的邻接矩阵不是二值矩阵，每个元素均介于0-1之间
    return topk_adj(rec_adj, e * 2)  # topk_adj 函数将解码器输出的邻接矩阵转换为二值矩阵，详见utils.py


def eval(adj, base_adj=None):
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)

    adj_ -= np.diag(np.diag(adj_))
    gr = nx.from_numpy_array(adj_)
    assert ((adj_ == adj_.T).all())

    d = compute_graph_statistics(adj_)
    pprint(d)

    if base_adj is not None:
        # base_adj = base_adj.numpy()
        base_gr = nx.from_numpy_array(base_adj)
        bd = compute_graph_statistics(base_adj)
        diff_d = {}

        for k in list(d.keys()):
            diff_d[k] = round(abs(d[k] - bd[k]), 4)
    return diff_d
