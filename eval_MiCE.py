from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import argparse
import time
from torchvision import transforms, datasets
from dataset import *
from torch.utils.data import ConcatDataset
from util import *

from ELBO import MiCE_ELBO
from sklearn import metrics

def parse_option():
    parser = argparse.ArgumentParser('argument for eval')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar20', 'stl10'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet34_cifar', choices=['resnet34', 'resnet34_cifar'])
    parser.add_argument('--low_dim', type=int, default=128, help='Dimension of each embedding')
    parser.add_argument('--test_path', type=str, default=None, help='the model to test')

    # loss function
    parser.add_argument('--nu', type=int, default=16384, help='Dimension of the queue')
    parser.add_argument('--tau', type=float, default=1.0, help='temperature')

    # EMA setting
    parser.add_argument('--m', type=float, default=0.999, help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--suffix', type=str, default=None, help='suffix to add on the model name')
    parser.add_argument('--data_folder', type=str, default="./", help='path to data')
    parser.add_argument('--save_path', type=str, default="./model_save/eval", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="./tensorboard/eval", help='path to tensorboard')

    opt = parser.parse_args()

    opt.model_name = opt.test_path.split('/')[-2]
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10' or opt.dataset == 'stl10':
        opt.n_class = 10
    elif opt.dataset == 'cifar20':
        opt.n_class = 20
    elif opt.dataset == 'imagenet_dog':
        opt.n_class = 15

    return opt


def get_MiCE_performance(model, model_ema, elbo, dataloader, ndata, n_class, batchSize=100):
    model.eval()
    model_ema.eval()
    all_cluster_labels = []
    all_true_labels = []
    all_pi_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            x1, x2, x3 = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()
            targets = targets.cuda()

            with torch.no_grad():
                f = model(x1)
                v = model_ema(x2)
            _, g = model(x3, True)

            loss, out, posterior, pi_logit = elbo(f, v, g)

            feat_pseudo_label = torch.argmax(posterior, dim=1)  # batch x 1
            pi_label = torch.argmax(pi_logit, dim=1)
            all_cluster_labels.extend(list(feat_pseudo_label.data.cpu().numpy()))
            all_true_labels.extend(list(targets.data.cpu().numpy()))
            all_pi_labels.extend(list(pi_label.data.cpu().numpy()))

    all_true_labels = np.array(all_true_labels)
    all_cluster_labels = np.array(all_cluster_labels)
    all_pi_labels = np.array(all_pi_labels)

    print("True label stat:", np.bincount(all_true_labels.astype(int)))
    print("Cluster label stat:", np.bincount(all_cluster_labels.astype(int)))

    mice_acc = acc(all_true_labels, all_cluster_labels)
    pi_cluster_acc = acc(all_true_labels, all_pi_labels)

    nmi = metrics.normalized_mutual_info_score(labels_true=all_true_labels, labels_pred=all_cluster_labels)
    ari = metrics.adjusted_rand_score(labels_true=all_true_labels, labels_pred=all_cluster_labels)
    return mice_acc, pi_cluster_acc, nmi, ari


def main():
    args = parse_option()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    image_size, mean, std, n_class = get_dataset_stat(args.dataset)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset, test_dataset = create_dataset(args.dataset, train_transform, train_transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    n_full_data = len(full_dataset)
    print(n_full_data)
    full_loader = torch.utils.data.DataLoader(full_dataset,
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    model, model_ema = create_model(args.model, n_class)
    elbo = MiCE_ELBO(args.low_dim, n_full_data, args.nu, tau=args.tau, n_class=n_class).cuda(
        args.gpu)

    ckpt = torch.load(args.test_path)
    model.load_state_dict(ckpt['model'])

    model_ema.load_state_dict(ckpt['model_ema'])
    elbo.load_state_dict(ckpt['elbo'])

    print("==> loaded checkpoint '{}' (epoch {})".format(args.test_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    model_ema = model_ema.cuda()
    model.eval()
    model_ema.eval()
    cudnn.benchmark = True

    n_data = len(full_dataset)
    mice_acc, mice_pi_acc, mice_nmi, mice_ari = get_MiCE_performance(
                                                model, model_ema, elbo, full_loader, n_data, n_class
    )
    print("CMoE average:", " NMI:", mice_nmi, "| Cluster ACC:", mice_acc, "| ARI:", mice_ari)

if __name__ == '__main__':
    main()
