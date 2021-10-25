import pickle
import warnings
import sklearn.metrics as metrics
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch.nn import init
import pandas as pd
import os
import numpy as np
import torch
import random
from graph_sample import datasets1
from sklearn.model_selection import StratifiedKFold
from neuroCombat import neuroCombat
warnings.filterwarnings("ignore")
# 导入数据阶段
dim = 200
path1 = 'cc200/ABIDE_pcp/cpac/filt_global/'
path2 = 'cc200/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'

Site_name = {
    'CALTECH': 0,
    'CMU': 1,
    'KKI': 2,
    'LEUVEN_1': 3,
    'LEUVEN_2': 3,
    'MAX_MUN': 4,
    'NYU': 5,
    'OHSU': 6,
    'OLIN': 7,
    'PITT': 8,
    'SBL': 9,
    'SDSU': 10,
    'STANFORD': 11,
    'TRINITY': 12,
    'UCLA_1': 13,
    'UCLA_2': 13,
    'UM_1': 14,
    'UM_2': 14,
    'USM': 15,
    'YALE': 16,
}


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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(path1, path2):
    profix = path1
    dirs = os.listdir(profix)
    all = {}
    labels = {}
    all_data = []
    label = []
    site = []
    gender = []
    for filename in dirs:
        a = np.loadtxt(path1 + filename)
        print(filename)
        a = a.transpose()
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
                site.append(Site_name[data['SITE_ID'][i]])
                gender.append(data['SEX'][i])
                break
    label = np.array(label)
    site = np.array(site)
    gender = np.array(gender)
    np.save('./site.npy', site)
    np.save('./gender.npy', gender)
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

    new_array = np.zeros((19900, 871))  #
    # 取值上三角
    for (index, graph) in enumerate(data_array):
        v = 0
        for a in range(200):
            for b in range(200):
                if b > a:
                    new_array[v][index] = graph[a][b]
                    v += 1
    # 增加性别等变量
    np.save("./features.npy", new_array)
    data = np.load("/home/ubnn/PycharmProjects/GroupINN/features.npy")
    batch = np.load("/home/ubnn/PycharmProjects/GroupINN/site.npy")
    gender = np.load("/home/ubnn/PycharmProjects/GroupINN/gender.npy")
    covars = {'batch': batch.tolist(),  # site
              'gender': gender.tolist()}
    covars = pd.DataFrame(covars)
    categorical_cols = ['gender']
    batch_col = 'batch'
    # Harmonization step:
    data_combat = neuroCombat(dat=data, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
    data_array = []
    for i in range(871):
        data_array.append(to_adj(data_combat[:, i]))
    # print(data_array[0])
    # 转为矩阵
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3))

    return data_array


###############################################################################
###############################################################################
def to_adj(vector):
    size = 200
    x = np.zeros((size, size))
    c = 0
    for i in range(0, size):
        if i < size:
            for j in range(0, i):
                x[i][j] = vector[c]
                x[j][i] = vector[c]
                c = c + 1
    return x


#################################################################################
###############################################################################

# 数据集划分

def cross_val(A, labels):
    kf = StratifiedKFold(n_splits=10, random_state=100, shuffle=True)
    zip_list = list(zip(A, labels))
    random.Random(12).shuffle(zip_list)
    A, labels = zip(*zip_list)
    test_data_loader = []
    train_data_loader = []
    valid_data_loader = []
    A = np.array(A)
    labels = np.array(labels)
    for kk, (train_index, test_index) in enumerate(kf.split(A, labels)):
        print(kk, test_index)
        train_val_adj, test_adj = A[train_index], A[test_index]
        train_val_labels, test_labels = labels[train_index], labels[test_index]
        tv_folder = StratifiedKFold(n_splits=10, random_state=100, shuffle=True).split(train_val_adj, train_val_labels)
        for t_idx, v_idx in tv_folder:
            train_adj, train_labels = train_val_adj[t_idx], train_val_labels[t_idx]
            val_adj, val_labels = train_val_adj[v_idx], train_val_labels[v_idx]

        "Fold-保存数据集合"
        torch.save(train_val_adj, '/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(kk) + 'train_adj.pth')
        torch.save(train_val_labels, '/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(kk) + 'train_labels.pth')
        torch.save(val_adj, '/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(kk) + 'val_adj.pth')
        torch.save(val_labels, '/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(kk) + 'val_labels.pth')
        torch.save(test_adj, '/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(kk) + 'test_adj.pth')
        torch.save(test_labels, '/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(kk) + 'test_labels.pth')

        dataset_sampler = datasets1(test_adj, test_labels)
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        test_data_loader.append(test_dataset_loader)
        dataset_sampler = datasets1(train_val_adj, train_val_labels)
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        train_data_loader.append(train_dataset_loader)
        dataset_sampler = datasets1(val_adj, val_labels)
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        valid_data_loader.append(val_dataset_loader)

    return train_data_loader, valid_data_loader, test_data_loader


#############################################################################
##############################################################################

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

    def forward(self, A):

        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]
        p_reduce, p_SX = self.dim_reduce(s_feature_p, 10, 0.2, 0.3, 0.1, self.kernel_p)

        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce, n_SX = self.dim_reduce(s_feature_n, 10, 0.2, 0.5, 0.1, self.kernel_n)

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
    probs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            labels.append(data['label'].long().numpy())
            ypred, loss = model(adj)
            # loss = F.cross_entropy(ypred, label, size_average=True)
            # avg_loss += loss
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            probs.append(ypred[:, 1].cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * 32 > max_num_examples:
                    break
    avg_loss /= batch_idx + 1

    labels = np.hstack(labels)
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

    return avg_loss, result


result = [0, 0, 0, 0, 0]
ii = -1


def train(epoch_, dataset, model, optimizer, val_dataset=None, test_dataset=None, device='cpu'):
    for name in model.state_dict():
        print(name)
    iter = 0
    best_val_acc = 0.0
    bestVal = []
    best = 0
    global ii
    ii += 1
    for epoch_index in range(epoch_):
        avg_loss = 0.0
        model.train()
        print(epoch_index)
        for batch_idx, data in enumerate(dataset):
            for k, v in model.named_parameters():
                if k == 'kernel_p' or k == 'kernel_n':
                    pass
                v.requires_grad = True

            model.zero_grad()
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            pred, losses = model(adj)
            loss = F.cross_entropy(pred, label, size_average=True)
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.sum(torch.abs(param))
            loss += losses
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            iter += 1
            avg_loss += loss
        avg_loss /= batch_idx + 1
        train_loss, train_result = evaluate(dataset, model, name='Train', device=device)
        if val_dataset is not None:
            val_loss, val_result = evaluate(val_dataset, model, name='Validation', device=device)
            if (epoch + 1) % 1 == 0:
                print('train', train_result)
                print('val', val_result)
            if val_result['acc'] >= best_val_acc:
                torch.save(model.state_dict(), './models/checkpoint' + str(ii) + '.pt')
                torch.save(optimizer.state_dict(), './models/optimizer' + str(ii) + '.pt')
                best_val_acc = val_result['acc']
                bestVal = val_result
                best = epoch_index
    print(bestVal)
    print(best)
    model.load_state_dict(torch.load('./models/checkpoint' + str(ii) + '.pt'))
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
    global flag
    flag = 0
    print(Site_name)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: ', device)
    test_data_loaders = []
    train_data_loaders = []
    valid_data_loaders = []
    for i in range(0, 10):
        if os.path.exists('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'train_adj.pth'):
            "10折-保存数据集合"
            train_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'train_adj.pth')
            train_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'train_labels.pth')
            val_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'val_adj.pth')
            val_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'val_labels.pth')
            test_adj = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'test_adj.pth')
            test_labels = torch.load('/home/ubnn/PycharmProjects/GroupINN/10_fold/' + str(i) + 'test_labels.pth')

            dataset_sampler = datasets1(test_adj, test_labels)
            test_dataset_loader = torch.utils.data.DataLoader(
                dataset_sampler,
                batch_size=32,
                shuffle=False,
                num_workers=0)
            test_data_loaders.append(test_dataset_loader)
            dataset_sampler = datasets1(train_adj, train_labels)
            train_dataset_loader = torch.utils.data.DataLoader(
                dataset_sampler,
                batch_size=32,
                shuffle=False,
                num_workers=0)
            train_data_loaders.append(train_dataset_loader)
            dataset_sampler = datasets1(val_adj, val_labels)
            val_dataset_loader = torch.utils.data.DataLoader(
                dataset_sampler,
                batch_size=32,
                shuffle=False,
                num_workers=0)
            valid_data_loaders.append(val_dataset_loader)
            print(str(i) + '_fold loading finished')
        else:
            # 导入数据
            _, _, raw_data, labels = load_data(data_path, label_path)  # raw_data [871 116 ?]  labels [871]
            # 划分时间窗
            adj = cal_pcc(raw_data)
            print('finished')
            train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, labels)
    result = []
    for epoch in range(450, 600, 20):
        for i in range(len(train_data_loaders)):
            model = model_gnn(8, 8, 2)
            optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.001)
            model.to(device)
            print('gan_model:', model)
            model = train(epoch, train_data_loaders[i], model, optimizer2, val_dataset=valid_data_loaders[i], test_dataset=test_data_loaders[i], device=device)
            _, test_result = evaluate(test_data_loaders[i], model, name='Test', device=device)
            print(test_result)
            result.append(test_result)
            del model
            del test_result
            del optimizer2
            # 将fold从零开始
    i = 0
    ii = -1
    saveList(result, './result7.pickle')
    result.append(epoch)
    print(result)
