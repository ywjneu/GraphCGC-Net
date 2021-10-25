import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *
import warnings

warnings.filterwarnings("ignore")


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # in_features 是输入维度
        self.out_features = out_features  # out_features 是输出维度
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 权重矩阵 --- torch.nn.parameter 将一个不可训练的类型 Tensor 转换成可以训练的类型 parameter
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)  # register_parameter 的作用是向模型中添加参数
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  # 权重初始化
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:  # 偏差初始化
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # 矩阵相乘，input 是图的特征向量矩阵
        output = torch.matmul(adj, support)  # 矩阵稀疏相乘， adj 是图的邻接矩阵
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Encoder(nn.Module):
    def __init__(self, coarse_size, latent_gcn, latent2_gcn, latent_linear):
        super(Encoder, self).__init__()
        self.latent_linear = latent_linear
        self.latent_gcn = latent_gcn
        self.latent2_gcn = latent2_gcn
        self.coarse_size = coarse_size

        self.gc_p = GraphConvolution(coarse_size, latent_gcn)
        self.gc2_p = GraphConvolution(latent_gcn, latent2_gcn)
        self.mean_p = nn.Sequential(
            nn.Linear(latent_linear, latent_linear),
            nn.BatchNorm1d(latent_linear)
        )
        self.logstd_p = nn.Sequential(
            nn.Linear(latent_linear, latent_linear),
            nn.BatchNorm1d(latent_linear)
        )

        self.gc_n = GraphConvolution(coarse_size, latent_gcn)
        self.gc2_n = GraphConvolution(latent_gcn, latent2_gcn)
        self.mean_n = nn.Sequential(
            nn.Linear(latent_linear, latent_linear),
            nn.BatchNorm1d(latent_linear)
        )
        self.logstd_n = nn.Sequential(
            nn.Linear(latent_linear, latent_linear),
            nn.BatchNorm1d(latent_linear)
        )

    def forward(self, adj):
        x = torch.eye(8).cuda()
        adj_list = []
        reparametrized_noise = torch.randn([self.coarse_size, self.latent_gcn], requires_grad=True).cuda()
        for i in range(adj.size(0)):  # 样本个数
            sub_adj = []
            adj_p = adj[i]
            for ii in range(adj_p.size(0)):
                if ii == 0:
                    p_reduce = adj_p[ii]  # 8*8
                    p_reduce = normalize(p_reduce)
                    x1 = nn.LeakyReLU(0.2, inplace=True)(self.gc_p(x, p_reduce))
                    x2 = nn.LeakyReLU(0.2, inplace=True)(self.gc2_p(x1, p_reduce))
                    mean_p = self.mean_p(x2)
                    logvar_p = self.logstd_p(x2)
                    std_p = logvar_p.mul(0.5).exp_()
                    reparametrized_noise_p = mean_p + std_p * reparametrized_noise
                    sub_adj.append(reparametrized_noise_p.unsqueeze(0))
                else:
                    n_reduce = adj_p[ii]  # 8*8
                    n_reduce = normalize(n_reduce)
                    x1 = nn.LeakyReLU(0.2, inplace=True)(self.gc_n(x, n_reduce))
                    x2 = nn.LeakyReLU(0.2, inplace=True)(self.gc2_n(x1, n_reduce))
                    mean_n = self.mean_n(x2)
                    logvar_n = self.logstd_n(x2)
                    std_n = logvar_n.mul(0.5).exp_()
                    reparametrized_noise_n = mean_n + std_n * reparametrized_noise
                    sub_adj.append(reparametrized_noise_n.unsqueeze(0))
            concat_adj = torch.cat(sub_adj, 0)
            adj_list.append(concat_adj.unsqueeze(0))
        x_ = torch.cat(adj_list, 0)
        return x_


class codeDiscriminator(nn.Module):
    def __init__(self, coarse_size, latent_gcn, num_units):
        super(codeDiscriminator, self).__init__()
        self.code_size = coarse_size*latent_gcn*2  # negative and positive
        self.num_inits = num_units
        self.l1 = nn.Sequential(nn.Linear(self.code_size, num_units),
                                nn.LeakyReLU(0.2, inplace=True)
                                )
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.LeakyReLU(0.2, inplace=True)
                                )
        self.l3 = nn.Sequential(nn.Linear(num_units, 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.l1(x)
        h2 = self.l2(h1)
        output = self.l3(h2)
        return output


class Generator(nn.Module):

    def __init__(self, latent_gcn, latent2_linear, coarse_size):
        super(Generator, self).__init__()
        self.latent_gcn = latent_gcn
        self.latent2_linear = latent2_linear
        self.coarse_size = coarse_size

        self.layer1 = nn.Sequential(
            nn.Linear(latent_gcn, coarse_size, bias=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(coarse_size, latent2_linear, bias=True),
        )

        self.layer1_n = nn.Sequential(
            nn.Linear(latent_gcn, coarse_size, bias=True),
        )
        self.layer2_n = nn.Sequential(
            nn.Linear(coarse_size, latent2_linear, bias=True),
        )

    def forward(self, noise):
        graph = []
        for i in range(noise.size(0)):
            sub_graph = []
            x_graph = noise[i]
            for ii in range(x_graph.size(0)):
                if ii == 0:
                    x_sub_p = x_graph[ii]
                    x_l1 = self.layer1(x_sub_p)
                    x_l2 = self.layer2(x_l1)
                    predict = torch.mm(x_l2, x_l2.t())
                    predict = F.softplus(predict)
                    sub_graph.append(predict.unsqueeze(0))
                else:
                    x_sub_n = x_graph[ii]
                    x_l1_n = self.layer1_n(x_sub_n)
                    x_l2_n = self.layer2_n(x_l1_n)
                    predict_n = torch.mm(x_l2_n, x_l2_n.t())
                    predict_n = F.softplus(predict_n)
                    sub_graph.append(predict_n.unsqueeze(0))
            DoubleGraph = torch.cat(sub_graph, 0)
            graph.append(DoubleGraph.unsqueeze(0))
        x_double = torch.cat(graph, 0)
        return x_double


class Discriminator(nn.Module):
    def __init__(self, coarse_size, latent_gcn, latent2_gcn):
        super(Discriminator, self).__init__()
        self.coarse_size = coarse_size
        self.latent_gcn = latent_gcn
        self.latent2_gcn = latent2_gcn
        self.gc_p = GraphConvolution(coarse_size, latent_gcn)
        self.gc2_p = GraphConvolution(latent_gcn, latent2_gcn)
        self.gc_n = GraphConvolution(coarse_size, latent_gcn)
        self.gc2_n = GraphConvolution(latent_gcn, latent2_gcn)

        self.layer1 = nn.Sequential(
            nn.Linear(2*coarse_size*latent2_gcn, coarse_size*coarse_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(coarse_size*coarse_size, int(coarse_size*coarse_size*0.5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(coarse_size*coarse_size*0.5), 1)
        )

    def forward(self, adj):
        x = torch.eye(8).cuda()
        adj_list = []
        for i in range(adj.size(0)):
            sub_adj = []
            adj_p = adj[i]
            for ii in range(adj_p.size(0)):
                if ii == 0:
                    p_reduce = adj_p[ii]
                    p_reduce = normalize(p_reduce)
                    x1 = nn.LeakyReLU(0.2, inplace=True)(self.gc_p(x, p_reduce))
                    x2 = nn.LeakyReLU(0.2, inplace=True)(self.gc2_p(x1, p_reduce))
                    sub_adj.append(x2.unsqueeze(0))
                else:
                    n_reduce = adj_p[ii]
                    n_reduce = normalize(n_reduce)
                    x1 = nn.LeakyReLU(0.2, inplace=True)(self.gc_n(x, n_reduce))
                    x2 = nn.LeakyReLU(0.2, inplace=True)(self.gc2_n(x1, n_reduce))
                    sub_adj.append(x2.unsqueeze(0))
            concat_adj = torch.cat(sub_adj, 0)
            adj_list.append(concat_adj.unsqueeze(0))
        x_ = torch.cat(adj_list, 0)
        x = x_.reshape(x_.size(0), -1)
        x = self.layer1(x)

        return x

    def similarity(self, adj):
        x = torch.eye(8).cuda()
        adj_list = []
        for i in range(adj.size(0)):
            sub_adj = []
            adj_p = adj[i]
            for ii in range(adj_p.size(0)):
                if ii == 0:
                    p_reduce = adj_p[ii]
                    p_reduce = normalize(p_reduce)
                    x1 = nn.LeakyReLU(0.2, inplace=True)(self.gc_p(x, p_reduce))
                    x2 = nn.LeakyReLU(0.2, inplace=True)(self.gc2_p(x1, p_reduce))
                    sub_adj.append(x2.unsqueeze(0))
                else:
                    n_reduce = adj_p[ii]
                    n_reduce = normalize(n_reduce)
                    x1 = nn.LeakyReLU(0.2, inplace=True)(self.gc_n(x, n_reduce))
                    x2 = nn.LeakyReLU(0.2, inplace=True)(self.gc2_n(x1, n_reduce))
                    sub_adj.append(x2.unsqueeze(0))
            concat_adj = torch.cat(sub_adj, 0)
            adj_list.append(concat_adj.unsqueeze(0))
        x_ = torch.cat(adj_list, 0)
        return x_


def to_adj(vector):
    size = 8
    x = np.zeros((size, size))
    c = 0
    for i in range(1, size + 1):
        # diag
        x[i - 1][i - 1] = vector[c]
        c = c + 1
        if i < 8:
            for j in range(0, i):
                x[i][j] = vector[c]
                x[j][i] = vector[c]
                c = c + 1
    return x


def homeomorphicMap(start, end):
    Demo = []
    alphaOne = torch.rand(1, 1).cuda()
    oneVector = (alphaOne * start.data + (1 - alphaOne) * end.data).requires_grad_(True)
    alphaTwo = torch.rand(1, 1).cuda()
    twoVector = (alphaTwo * start.data + (1 - alphaTwo) * end.data).requires_grad_(True)

    Demo.append(oneVector.unsqueeze(0))
    Demo.append(twoVector.unsqueeze(0))
    Vector = torch.cat(Demo, dim=0)
    return Vector
