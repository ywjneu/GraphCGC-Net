import pickle
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from gan.models import *
from gan.options import Options
import warnings
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch.nn import init
import pandas as pd
import os
import numpy as np
import random
from graph_sample import datasets1

warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


opt = Options().initialize()

z_out_size = 80
E = Encoder(coarse_size=opt.coarse_size, latent_gcn=opt.latent_gcn, latent2_gcn=opt.latent2_gcn, latent_linear=opt.latent_linear).cuda()

G_normal = Generator(coarse_size=opt.latent_gcn, latent_gcn=opt.latent_gcn, latent2_linear=opt.latent2_gcn).cuda()

G_asd = Generator(coarse_size=opt.latent_gcn, latent_gcn=opt.latent_gcn, latent2_linear=opt.latent2_gcn).cuda()
###############################################################
###############################################################
# 导入数据阶段
dim = 200
path1 = 'cc200/ABIDE_pcp/cpac/filt_global/'
path2 = 'cc200/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'


def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key


'''
导入数据进来
'''


def load_data(path1, path2):
    profix = path1
    dirs = os.listdir(profix)
    all = {}
    labels = {}
    all_data = []
    label = []
    files = open('files.txt', 'r')
    for filename in dirs:
        filename = files.readline().strip()
        a = np.loadtxt(path1 + filename)
        print(filename)
        a = a.transpose()
        # a = a.tolist()
        all[filename] = a
        all_data.append(a)
        data = pd.read_csv(path2)
        for i in range(len(data)):
            if get_key(filename) == data['FILE_ID'][i]:
                if int(data['DX_GROUP'][i]) == 1:
                    labels[filename] = int(data['DX_GROUP'][i])
                    label.append(int(data['DX_GROUP'][i]))
                else:
                    labels[filename] = 0
                    label.append(0)
                break
    label = np.array(label)
    return all, labels, all_data, label  # 871 * 116 * ?


def cal_pcc(data):
    '''
    :param data:  图   871 * 116 * ?
    :return:  adj
    '''
    corr_matrix = []
    for key in range(len(data)):  # 每一个sample
        corr_mat = np.corrcoef(data[key])
        # if key == 5:
        #    print(corr_mat)
        corr_mat = np.arctanh(corr_mat - np.eye(corr_mat.shape[0]))

        corr_matrix.append(corr_mat)
    data_array = np.array(corr_matrix)  # 871 116 116
    where_are_nan = np.isnan(data_array)  # 找出数据中为nan的
    where_are_inf = np.isinf(data_array)  # 找出数据中为inf的
    for bb in range(0, data_len):
        for i in range(0, dim):
            for j in range(0, dim):
                if where_are_nan[bb][i][j]:
                    data_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    data_array[bb][i][j] = 0.8
    # print(data_array[0])
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3))

    return data_array


###############################################################################
###############################################################################


#################################################################################
###############################################################################

# 数据集划分

def cross_val(data_k):
    test_data_loader = []
    train_data_loader = []
    valid_data_loader = []

    "五折-超点数据读取"
    train_adj = torch.load(
        '/home/ubnn/PycharmProjects/GroupINN/supernode_data/' + str(data_k) + '_' + 'train_superNode.pth')
    train_labels = torch.load(
        '/home/ubnn/PycharmProjects/GroupINN/supernode_data/' + str(data_k) + '_' + 'superNode_label_.pth')

    val_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/supernode_data/' + str(data_k) + '_' + 'val_superNode.pth')
    val_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/supernode_data/' + str(data_k) + '_' + 'val_label_.pth')
    test_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/supernode_data/' + str(data_k) + '_' + 'test_superNode.pth')
    test_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/supernode_data/' + str(data_k) + '_' + 'test_label_.pth')

    # # =========================================================
    """# 加载数据集的时候，将正负分开。"""
    positive = []
    positive_label = []
    negative = []
    negative_label = []
    for i, j in enumerate(train_labels):
        if j == 0:
            positive.append(train_adj[i].unsqueeze(0))
            positive_label.append(j.unsqueeze(0))
        else:
            negative.append(train_adj[i].unsqueeze(0))
            negative_label.append(j.unsqueeze(0))
    # # =========================================================
    normal_model = 200
    G_normal.load_state_dict( torch.load('./gan_model/checkpoint/normal/G_noW_epoch' + str(normal_model) + '.pth'))
    # 选取训练集生成
    # 选取训练集生成
    # 分割
    demo = torch.cat(positive, dim=0)
    demo_label = torch.cat(positive_label, dim=0)
    # # =========================================================
    generator_tc = []
    new_labels_tc = []
    dataset_sampler_temp = datasets1(demo, demo_label)
    dataset_loader_normal = torch.utils.data.DataLoader(dataset_sampler_temp, batch_size=2, shuffle=False,num_workers=0)
    # 生成一倍正常人
    for i, data in enumerate(dataset_loader_normal):
        g_label = Variable(torch.tensor([0, 0]))
        noisev = torch.randn((2, 2, 8, 5), requires_grad=False).cuda()
        rec_noise = G_normal(noisev)
        generator_tc.append(rec_noise)
        new_labels_tc.append(g_label)
        print(i)
    generator_tc = torch.cat(generator_tc, 0)
    new_labels_tc = torch.cat(new_labels_tc, 0)
    # =========================================================
    asd_model = 200
    G_normal.load_state_dict(torch.load('./gan_model/checkpoint/ASD/G_noW_epoch' + str(asd_model) + '.pth'))
    # 选取训练集生成
    demo_asd = torch.cat(negative, dim=0)
    demo_label_asd = torch.cat(negative_label, dim=0)
    generator_asd = []
    new_labels_asd = []
    dataset_sampler_asd = datasets1(demo_asd, demo_label_asd)
    dataset_loader_asd = torch.utils.data.DataLoader(dataset_sampler_asd, batch_size=2, shuffle=False, num_workers=0)
    # 生产ASD
    for i, data in enumerate(dataset_loader_asd):
        g_label = Variable(torch.tensor([1, 1]))
        noisev = torch.randn((2, 2, 8, 5), requires_grad=False).cuda()
        rec_noise = G_asd(noisev)
        generator_asd.append(rec_noise)
        new_labels_asd.append(g_label)
        print(i)
    generator_asd = torch.cat(generator_asd, 0)
    new_labels_asd = torch.cat(new_labels_asd, 0)
    # 将生成的正负混合
    generator = torch.cat([generator_asd, generator_tc], dim=0)
    new_labels = torch.cat([new_labels_asd, new_labels_tc], dim=0)

    generator_dateset_sampler = datasets1(generator, new_labels)
    # =========================================================
    # Train
    train_dataset_sampler = datasets1(train_adj, train_labels)
    data_total = torch.utils.data.ConcatDataset([train_dataset_sampler, generator_dateset_sampler])
    train_dataset_loader = torch.utils.data.DataLoader(data_total, batch_size=32, shuffle=True, num_workers=0)
    train_data_loader.append(train_dataset_loader)
    # Test
    test_dataset_sampler = datasets1(test_adj, test_labels)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset_sampler, batch_size=32, shuffle=True, num_workers=0)
    test_data_loader.append(test_dataset_loader)
    # Val
    val_dataset_sampler = datasets1(val_adj, val_labels)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset_sampler, batch_size=32, shuffle=True, num_workers=0)
    valid_data_loader.append(val_dataset_loader)

    return train_data_loader, valid_data_loader, test_data_loader


#############################################################################
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
        eye = torch.eye(feature_dim).to(device)  # 生成对角矩阵 feature_dim * feature_dim
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

        self.lin1 = nn.Linear(2 * 8 * in_dim, 16)
        self.Dropout = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(16, self.out_dim)

        self.losses = []
        self.reset_weigths()

    def dim_reduce(self, adj_matrix, num_reduce, ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
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

    def forward(self, A, flag):

        A = torch.transpose(A, 1, 0)
        p_reduce = A[0]
        n_reduce = A[1]
        if flag is True:
            p_reduce, p_SX = self.dim_reduce(p_reduce, 10, 0.2, 0.3, 0.1, self.kernel_p)

        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        if flag is True:
            n_reduce, n_SX = self.dim_reduce(n_reduce, 10, 0.2, 0.5, 0.1, self.kernel_n)
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        n_conv3 = self.gcn3_n(n_conv2, n_reduce)

        conv_concat = torch.cat([p_conv3, n_conv3], -1).reshape([-1, 2 * 8 * 8])

        output = self.lin2(self.Dropout(self.lin1(conv_concat)))
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return output, loss

    def consist_loss(self, s):
        s = s.squeeze()
        if len(s) == 0:
            return 0
        else:
            s = torch.sigmoid(s)
            W = torch.ones(s.shape[0], s.shape[0])
            D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
            L = D - W
            L = L.to(device)
            res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
            return res


def evaluate(dataset, model, name='Validation', max_num_examples=None, device='cpu'):
    model.eval()
    avg_loss = 0.0
    preds = []
    labels = []
    roc_labels = []
    probs = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            labels.append(data['label'].long().numpy())
            ypred, loss = model(adj, flag=False)
            _, indices = torch.max(ypred, 1)
            roc_labels.append(ypred.cpu().data.numpy()[:, 1])
            preds.append(indices.cpu().data.numpy())
            probs.append(ypred[:, 1].cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * 32 > max_num_examples:
                    break
    avg_loss /= batch_idx + 1

    labels = np.hstack(labels)
    roc_labels = np.hstack(roc_labels)
    preds = np.hstack(preds)
    probs = np.hstack(probs)
    # print(labels)
    global xx
    global yy
    from sklearn.metrics import confusion_matrix
    auc = metrics.roc_auc_score(labels, probs, sample_weight=None)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average='macro'),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds)}
    xx = preds
    yy = labels

    return roc_labels, labels, result


result = [0, 0, 0, 0, 0]


def train(dataset, model, val_dataset=None, test_dataset=None, device='cpu'):
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.001)
    optimizer2.load_state_dict(torch.load('./models/optimizer' + str(k_fold) + '.pt'))
    for name in model.state_dict():
        print(name)
    iter = 0
    best_val_acc = 0.0
    bestVal = []
    best = 0
    for epoch in range(300):
        avg_loss = 0.0
        model.train()
        print(epoch)
        for batch_idx, data in enumerate(dataset):
            if epoch < 0:
                for k, v in model.named_parameters():
                    if k != 'gcn1_p.kernel' and k != 'gcn2_p.kernel' and k != 'gcn3_p.kernel' and k != 'gcn1_n.kernel' and k != 'gcn2_n.kernel' and k != 'gcn3_n.kernel':
                        v.requires_grad = False  #
                model.zero_grad()
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                pred, losses = model(adj, flag=False)
                loss = F.cross_entropy(pred, label, size_average=True)
                loss += losses
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                iter += 1
                avg_loss += loss
            else:
                for k, v in model.named_parameters():
                    if k == 'kernel_p' or k == 'kernel_n':
                        pass
                    v.requires_grad = True

                model.zero_grad()
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                # print(timeseries)
                pred, losses = model(adj, flag=False)
                # print(pred)
                loss = F.cross_entropy(pred, label, size_average=True)
                l2_loss = 0
                for param in model.parameters():
                    l2_loss += torch.sum(torch.abs(param))

                loss += losses
                # loss = loss + 0.0001*l2_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer2.step()
                iter += 1
                avg_loss += loss
        avg_loss /= batch_idx + 1
        train_loss, _, train_result = evaluate(dataset, model, name='Train', device=device)
        if val_dataset is not None:
            val_loss, _, val_result = evaluate(val_dataset, model, name='Validation', device=device)
            if (epoch + 1) % 1 == 0:
                print('train', train_result)
                print('val', val_result)
            if val_result['acc'] >= best_val_acc:
                torch.save(model.state_dict(), './test_models/checkpoint' + str(k_fold) + '.pt')
                best_val_acc = val_result['acc']
                bestVal = val_result
                best = epoch
    print(bestVal)
    print(best)
    model.load_state_dict(torch.load('./test_models/checkpoint' + str(k_fold) + '.pt'))
    return model


###########################################################################################
###########################################################################################

# 主函数


def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent


global flag
if __name__ == "__main__":
    data_len = 871
    data_path = 'cc200/ABIDE_pcp/cpac/filt_global/'
    label_path = 'cc200/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
    # 设置种子
    set_seed(1)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: ', device)
    global k_fold
    # ---------w
    k_fold = ""
    # --------
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(k_fold)
    result = []
    for i in range(len(train_data_loaders)):
        model = model_gnn(8, 8, 2).cuda()
        model_argument = model_gnn(8, 8, 2).cuda()
        model.load_state_dict(torch.load('./models/checkpoint' + str(k_fold) + '.pt'))
        model_argument.load_state_dict(torch.load('./models/checkpoint' + str(k_fold) + '.pt'))

        model.cuda()
        print('gan_model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device)
        preds, labels, test_result = evaluate(test_data_loaders[i], model, name='Test', device=device)
        print(test_result)
        result.append(test_result)
        del model
        del test_result
    saveList(result, '../result7.pickle')
    print(result)
