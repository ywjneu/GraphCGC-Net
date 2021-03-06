import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import torch
dim = 200


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, neg_penalty):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 输入的维度
        self.out_dim = out_dim  # 输出的维度
        self.neg_penalty = neg_penalty  # 负值
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):

        # GCN-node

        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # 生成对角矩阵 feature_dim * feature_dim
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        I_cAXW = eye + self.c * AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        output = torch.nn.functional.softplus(y_norm)
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        return output


class model_gnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gcn1_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn3_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn3_n = GCN(in_dim, hidden_dim, 0.2)
        self.kernel_p = nn.Parameter(torch.FloatTensor(dim, 10))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(dim, 10))
        # print(self.kernel_p)
        # self.kernel_p = Variable(torch.randn(116, 5)).cuda()  # 116 5
        # self.kernel_n = Variable(torch.randn(116, 5)).cuda()   # 116 5
        self.lin1 = nn.Linear(2 * 100, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.losses = []
        self.reset_weigths()

    def dim_reduce(self, adj_matrix, num_reduce, ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(torch.tensordot(kernel_p, AF, [[0], [1]]), 1, 0)  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if ortho_penalty != 0:
            ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
            ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
            self.losses.append(ortho_loss)

        if variance_penalty != 0:
            variance = diag_elements.var()
            variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
            self.losses.append(variance_loss)

        if neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                      torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
            self.losses.append(neg_loss)
        self.losses.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    def reset_weigths(self):
        """reset weights
            """
        stdv = 1.0 / math.sqrt(dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, A, flag):
        A = torch.transpose(A, 1, 0)
        p_reduce = A[0]
        n_reduce = A[1]

        X = torch.eye(5).cuda()
        if flag is True:
            p_reduce = self.dim_reduce(p_reduce, 10, 0.2, 0.3, 0.1, self.kernel_p)
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        if flag is True:
            n_reduce = self.dim_reduce(n_reduce, 10, 0.2, 0.5, 0.1, self.kernel_n)
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        n_conv3 = self.gcn3_n(n_conv2, n_reduce)

        conv_concat = torch.cat([p_conv3, n_conv3], -1).reshape([-1, 200])
        output = self.lin2(self.lin1(conv_concat))
        # output = torch.softmax(output, dim=1)
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return output, loss
