import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser("Training Parser")

    def initialize(self):
        parser = self.parser

        parser.add_argument('--subjectType', type=int, default='', help="Normal=0 or ASD=1")
        parser.add_argument('--coarse_size', type=int, default='', help="coarsened graph size")
        parser.add_argument('--latent_gcn', type=int, default='', help="GCN latents size")
        parser.add_argument('--latent2_gcn', type=int, default='', help="GCN latents size")
        parser.add_argument('--latent_linear', type=int, default='', help="Linear latents size one")
        parser.add_argument('--latent2_linear', type=int, default='', help="Linear latents size two")
        parser.add_argument('--num_units', type=int, default='', help="Linear size")

        parser.add_argument('--adj_threshold', type=float, default='', help='threshold of adj edges')
        parser.add_argument('--max_epochs', type=int, default='', help='max epochs')

        parser.add_argument('--lr_E', type=float, default='', help='learning rate')
        parser.add_argument('--lr_CD', type=float, default='', help='learning rate')
        parser.add_argument('--lr_G', type=float, default='', help='learning rate')
        parser.add_argument('--lr_D', type=float, default='', help='learning rate')

        parser.add_argument('--beta', type=float, default='', help='beta')
        parser.add_argument('--alpha', type=float, default='', help='alpha')
        parser.add_argument('--gamma', type=float, default='', help='gamma')

        # ======================================================================
        parser.add_argument('--gpu', type=str, default='1', help='gpu id')
        parser.add_argument('--DATA_DIR', type=str, default='./coarsened_data/', help='output dir')

        parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
        parser.add_argument('--workers', type=int, default=4, help='num_workers')

        parser.add_argument('--model_dir', default='./model/', help='model dir')
        parser.add_argument('--clip', dest='clip', type=float, default=2.0, help='Gradient clipping.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

        opt = parser.parse_args()
        return opt
