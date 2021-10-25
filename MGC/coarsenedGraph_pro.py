import warnings
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch.nn import init
import torch
from graph_sample import dataset_pro

warnings.filterwarnings("ignore")
###############################################################
dim = 200
path1 = 'cc200/ABIDE_pcp/cpac/filt_global/'
path2 = 'cc200/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
data_path = '/home/ubnn/PycharmProjects/GroupINN/GAN_original/'


# 模型定义
# A 和 B 的哈达玛乘积可以写作 A*B
# 矩阵运算则写作torch.matmul(A,B)
# GCN_Layer 书写

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
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # 生成对角矩阵 feature_dim * feature_dim
        W = self.kernel
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        I_cAXW = eye + self.c * AXW
        # I_cAXW = self.c*AXW
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

        self.gcn1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn3 = GCN(in_dim, hidden_dim, 0.2)
        self.kernel_p = nn.Parameter(torch.FloatTensor(dim, 8))
        self.kernel_n = nn.Parameter(torch.FloatTensor(dim, 8))
        self.kernel = nn.Parameter(torch.FloatTensor(dim, 5))
        # print(self.kernel_p)
        # self.kernel_p = Variable(torch.randn(116, 5)).cuda()  # 116 5
        # self.kernel_n = Variable(torch.randn(116, 5)).cuda()   # 116 5
        self.lin1 = nn.Linear(2 * 8 * in_dim, 16)
        self.Dropout = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(16, self.out_dim)

        self.losses = []
        self.reset_weigths()

    def dim_reduce(self, adj_matrix, num_reduce, ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1, 0)  # num_reduce*batch*num_reduce*num_reduce
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
        return reduced_adj_matrix, None

    def reset_weigths(self):
        """reset weights
            """
        stdv = 1.0 / math.sqrt(dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, A):

        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]
        p_reduce, p_SX = self.dim_reduce(s_feature_p, 10, 0.2, 0.3, 0.1, self.kernel_p)
        # p_conv1 = self.gcn1_p(None, p_reduce)
        # p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        # print(p_SX)
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce, n_SX = self.dim_reduce(s_feature_n, 10, 0.2, 0.5, 0.1, self.kernel_n)
        # n_conv1 = self.gcn1_n(None, n_reduce)
        # n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        n_conv3 = self.gcn3_n(n_conv2, n_reduce)

        conv_concat = torch.cat([p_conv3, n_conv3], -1).reshape([-1, 2 * 8 * 8])

        # conv_concat = torch.cat([conv_concat, LSTM_out], -1)
        # print(conv_concat.shape)
        output = self.lin2(self.Dropout(self.lin1(conv_concat)))
        # output = torch.softmax(output, dim=1)
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return output, loss, p_reduce, n_reduce

    def consist_loss(self, s):
        s = s.squeeze()
        if len(s) == 0:
            return 0
        else:
            s = torch.sigmoid(s)
            W = torch.ones(s.shape[0], s.shape[0])
            D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
            L = D - W
            L = L.cuda()
            res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
            return res


def main():
    # 加载模型
    model = model_gnn(8, 8, 2).cuda()
    # ------
    i = 9
    # ------
    model.load_state_dict(torch.load('./final_models/10/checkpoint' + str(i) + '.pt'))

    "fold-数据读取"
    train_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'train_adj.pth')
    train_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'train_labels.pth')
    val_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'val_adj.pth')
    val_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'val_labels.pth')
    test_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'test_adj.pth')
    test_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'test_labels.pth')

    # adj labels 是数据和标签
    train_sampler = dataset_pro(train_adj, train_labels)
    test_sampler = dataset_pro(test_adj, test_labels)
    val_sampler = dataset_pro(val_adj, val_labels)
    train_dataloader = torch.utils.data.DataLoader(train_sampler, batch_size=32, shuffle=False)
    superNode_label_ = []
    train_superNode = []
    for data, labels in train_dataloader:
        a, b, p_reduce, n_reduce = model(Variable(data.to(torch.float32)).cuda())
        # 将原始数据处理完毕以后，
        p_ = p_reduce.unsqueeze(1)
        n_ = n_reduce.unsqueeze(1)
        superNode_label_.append(labels)
        data_cat = torch.cat((p_, n_), dim=1)
        train_superNode.append(data_cat)
    train_superNode = torch.cat(train_superNode, dim=0)
    superNode_label_ = torch.cat(superNode_label_, dim=0)

    torch.save(train_superNode, '/home/ubnn/PycharmProjects/GroupINN/supernode_data/'+str(i)+'_'+'train_superNode.pth')
    torch.save(superNode_label_, '/home/ubnn/PycharmProjects/GroupINN/supernode_data/'+str(i)+'_'+'superNode_label_.pth')

    test_dataloader = torch.utils.data.DataLoader(test_sampler, batch_size=32, shuffle=False)
    test_label_ = []
    test_superNode = []
    for data, labels in test_dataloader:
        a, b, p_reduce, n_reduce = model(Variable(data.to(torch.float32)).cuda())
        # 将原始数据处理完毕以后，
        p_ = p_reduce.unsqueeze(1)
        n_ = n_reduce.unsqueeze(1)
        test_label_.append(labels)
        data_cat = torch.cat((p_, n_), dim=1)
        test_superNode.append(data_cat)
    test_superNode = torch.cat(test_superNode, dim=0)
    test_label_ = torch.cat(test_label_, dim=0)

    torch.save(test_superNode, '/home/ubnn/PycharmProjects/GroupINN/supernode_data/'+str(i)+'_'+'test_superNode.pth')
    torch.save(test_label_, '/home/ubnn/PycharmProjects/GroupINN/supernode_data/'+str(i)+'_'+'test_label_.pth')

    val_dataloader = torch.utils.data.DataLoader(val_sampler, batch_size=32, shuffle=False)
    val_label_ = []
    val_superNode = []
    for data, labels in val_dataloader:
        a, b, p_reduce, n_reduce = model(Variable(data.to(torch.float32)).cuda())
        # 将原始数据处理完毕以后，
        p_ = p_reduce.unsqueeze(1)
        n_ = n_reduce.unsqueeze(1)
        val_label_.append(labels)
        data_cat = torch.cat((p_, n_), dim=1)
        val_superNode.append(data_cat)
    val_superNode = torch.cat(val_superNode, dim=0)
    val_label_ = torch.cat(val_label_, dim=0)

    torch.save(val_superNode, '/home/ubnn/PycharmProjects/GroupINN/supernode_data/'+str(i)+'_'+'val_superNode.pth')
    torch.save(val_label_, '/home/ubnn/PycharmProjects/GroupINN/supernode_data/'+str(i)+'_'+'val_label_.pth')


if __name__ == "__main__":
    main()
