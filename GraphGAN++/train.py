import random
from torch.utils.tensorboard import SummaryWriter
from methods import *
from dataset import *
from sklearn.metrics import mean_absolute_error
from utils import *
import centrality as central

warnings.filterwarnings("ignore")
opt = Options().initialize()
LAMBDA = 10
seed = 123


def set_seed(data):
    torch.manual_seed(data)
    torch.cuda.manual_seed_all(data)
    np.random.seed(data)
    random.seed(data)
    torch.backends.cudnn.deterministic = True


def calc_gradient_penalty(netD, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size() == x_gen.size(), "real and sampled sizes do not match"
    BATCH_SIZE, C, H, W = x.shape
    alpha = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).cuda()
    interpolates = alpha * x.data + (1 - alpha) * x_gen.data
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True,
                                    retain_graph=True, allow_unused=True)[0]
    gradients = Variable(gradients, requires_grad=True)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.data.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def train(dataset_loader):
    max_epochs = opt.max_epochs
    for epoch in range(max_epochs):
        EG_list = []
        D_real_list = []
        D_rec_list = []
        CD_list = []
        WGAN_list = []
        Rec_list = []

        for i, (data, _) in enumerate(dataset_loader):

            ###############################################
            # Train Encoder
            ###############################################
            for p in D.parameters():
                p.requires_grad = False
            for p in CD.parameters():
                p.requires_grad = False
            for p in E.parameters():
                p.requires_grad = True
            for p in G.parameters():
                p.requires_grad = True
            # 全局
            noisev = torch.randn((opt.batch_size, 2, opt.coarse_size, opt.latent_gcn), requires_grad=False).cuda()
            adj = Variable(data, requires_grad=False).cuda()
            for iters in range(1):
                e_optimizer.zero_grad()
                g_optimizer.zero_grad()
                code = E(adj)

                homeomorphic = homeomorphicMap(code[0], code[1])
                homeomorphic = Variable(homeomorphic, requires_grad=True).cuda()

                rec_adj = G(code)
                noise_adj = G(noisev)

                code_loss = -CD(code).mean()
                CD_list.append(code_loss)

                d_fake_loss = D(noise_adj)
                Fake_loss = - torch.mean(d_fake_loss)
                d_rec_loss = D(rec_adj)
                Rec_loss = - torch.mean(d_rec_loss)
                d_loss = -Fake_loss - Rec_loss

                # Rec loss
                Rec_loss_s = opt.beta * criterion_l1(rec_adj, adj)
                Gen_loss_s = opt.beta * criterion_l1(noise_adj, adj)

                threshold = opt.adj_threshold

                n_real_topology_, p_real_topology_ = central.topological_measures(adj, threshold)
                n_rec_topology_, p_rec_topology_ = central.topological_measures(rec_adj, threshold)
                # 0:closeness centrality    1:betweeness centrality    2:degree centrality
                n_Rectopology = mean_absolute_error(n_real_topology_[0], p_real_topology_[0])
                p_Rectopology = mean_absolute_error(n_rec_topology_[0], p_rec_topology_[0])
                rec_local_topology = (n_Rectopology + p_Rectopology)

                #  ================================================
                n_real_topology, p_real_topology = central.topological_measures(adj, threshold)
                n_fake_topology, p_fake_topology = central.topological_measures(noise_adj, threshold)
                # 0:closeness centrality    1:betweeness centrality    2:degree centrality
                n_topology = mean_absolute_error(n_fake_topology[0], n_real_topology[0])
                p_topology = mean_absolute_error(p_fake_topology[0], p_real_topology[0])
                noise_local_topology = (n_topology + p_topology)

                similarity_rec_enc = D.similarity(rec_adj)
                similarity_data = D.similarity(adj)
                rec_loss_m = criterion_l1(similarity_rec_enc, similarity_data)

                map_adj = G(homeomorphic)
                d_map_loss = D(map_adj)
                homeomorphic_loss = - torch.mean(d_map_loss)

                lossEG = (Rec_loss_s + Gen_loss_s + rec_loss_m) + code_loss + d_loss + (rec_local_topology + noise_local_topology) + homeomorphic_loss
                EG_list.append(lossEG)

                WGAN_list.append(d_loss)
                Rec_list.append(rec_loss_m)
                lossEG.backward()
                e_optimizer.step()
                g_optimizer.step()

            ###############################################
            # Train D
            ###############################################
            for p in D.parameters():
                p.requires_grad = True
            for p in CD.parameters():
                p.requires_grad = False
            for p in E.parameters():
                p.requires_grad = False
            for p in G.parameters():
                p.requires_grad = False

            for iters in range(1):
                d_optimizer.zero_grad()
                code = E(adj)
                rec_adj = G(code)
                noise_adj = G(noisev)
                map_adj = G(homeomorphic)

                real_valid = D(adj)
                rec_valid = D(rec_adj)
                fake_valid = D(noise_adj)
                map_valid = D(map_adj)

                gradient_penalty_real = calc_gradient_penalty(D, adj, noise_adj)
                gradient_penalty_re = calc_gradient_penalty(D, adj, rec_adj)
                gradient_penalty_mid = calc_gradient_penalty(D, adj, map_adj)

                loss_D_real = (torch.mean(fake_valid) - torch.mean(real_valid)) + opt.gamma * gradient_penalty_real
                loss_D_recon = (torch.mean(rec_valid) - torch.mean(real_valid)) + opt.gamma * gradient_penalty_re
                loss_D_mid = (torch.mean(map_valid) - torch.mean(real_valid)) + opt.gamma * gradient_penalty_mid

                loss_D = (loss_D_recon + loss_D_real + loss_D_mid)
                loss_D.backward()
                d_optimizer.step()
                D_real_list.append(loss_D_real)
                D_rec_list.append(loss_D_recon)

            ###############################################
            # Train CD
            ###############################################
            for p in D.parameters():
                p.requires_grad = False
            for p in CD.parameters():
                p.requires_grad = True
            for p in E.parameters():
                p.requires_grad = False
            for p in G.parameters():
                p.requires_grad = False
            for iters in range(1):
                cd_optimizer.zero_grad()
                code = E(adj)
                real_validity = CD(noisev)
                fake_validity = CD(code)
                gradient_penalty_cd = calc_gradient_penalty(CD, noisev, code)
                loss_CD = (torch.mean(fake_validity) - torch.mean(real_validity)) + gradient_penalty_cd
                loss_CD.backward()
                cd_optimizer.step()

        print('[{:d}/{:d}]: D_real:{:.4f}, D_enc:{:.4f},WGAN_list:{:.4f}, Loss_EG:{:.4f},CD_loss:{:.4f},Rec_list:{:.4f}'.format(epoch + 1, max_epochs,
                             torch.mean(torch.stack(D_real_list)), torch.mean(torch.stack(D_rec_list)),
                             torch.mean(torch.stack(WGAN_list)), torch.mean(torch.stack(EG_list)),
                             torch.mean(torch.stack(CD_list)),  torch.mean(torch.stack(Rec_list))))
        if (epoch + 1) % 1 == 0:
            torch.save(G.state_dict(), './{}/G_noW_epoch'.format(opt.model_dir+str(opt.subjectType)) + str(epoch + 1) + '.pth')
            torch.save(D.state_dict(), './{}/D_noW_epoch'.format(opt.model_dir+str(opt.subjectType)) + str(epoch + 1) + '.pth')
            torch.save(E.state_dict(), './{}/E_noW_epoch'.format(opt.model_dir+str(opt.subjectType)) + str(epoch + 1) + '.pth')
            torch.save(CD.state_dict(), './{}/CD_noW_epoch'.format(opt.model_dir+str(opt.subjectType)) + str(epoch + 1) + '.pth')


if __name__ == '__main__':
    print('=========== OPTIONS ===========')
    pprint(vars(opt))
    print(' ======== END OPTIONS ========\n\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # 0=normal  1=asd
    # -------------
    k_fold = ''
    # ----------------

    train_adj_mats, train_labels, = load_data(k_fold, opt.subjectType)
    dataset_sampler = DataSet(train_adj_mats, train_labels)
    dataset_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    train(dataset_loader)
