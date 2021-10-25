import torch.optim as optim
from models import *
from options import Options


opt = Options().initialize()

E = Encoder(coarse_size=opt.coarse_size, latent_gcn=opt.latent_gcn, latent2_gcn=opt.latent2_gcn, latent_linear=opt.latent_linear).cuda()
CD = codeDiscriminator(coarse_size=opt.coarse_size, latent_gcn=opt.latent_gcn, num_units=opt.num_units).cuda()
G = Generator(coarse_size=opt.latent_gcn, latent_gcn=opt.latent_gcn, latent2_linear=opt.latent2_gcn).cuda()
D = Discriminator(coarse_size=opt.coarse_size, latent_gcn=opt.latent_gcn, latent2_gcn=opt.latent2_gcn).cuda()

criterion_bce = nn.BCELoss()
criterion_bce.cuda()
criterion_l1 = nn.L1Loss()
criterion_l1.cuda()

loss_MSE = nn.MSELoss()
loss_MSE.cuda()

loss_BCE_logits = nn.BCEWithLogitsLoss()
loss_BCE_logits.cuda()

loss_BCE = nn.BCELoss()
loss_BCE.cuda()

g_optimizer = optim.Adam(G.parameters(), lr=opt.lr_G, betas=(0.5, 0.9))
d_optimizer = optim.Adam(D.parameters(), lr=opt.lr_D, betas=(0.5, 0.9))
e_optimizer = optim.Adam(E.parameters(), lr=opt.lr_E, betas=(0.5, 0.9))
cd_optimizer = optim.Adam(CD.parameters(), lr=opt.lr_CD, betas=(0.5, 0.9))
