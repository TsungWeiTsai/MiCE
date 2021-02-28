from __future__ import print_function
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import tensorboard_logger as tb_logger
from torchvision import transforms, datasets
from util import *
from ELBO import MiCE_ELBO
from dataset import get_dataset_stat, create_dataset
from torch.utils.data import ConcatDataset


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1.0, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='480,640,800', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar20', 'stl10'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet34_cifar', choices=['resnet34', 'resnet34_cifar'])
    parser.add_argument('--low_dim', type=int, default=128, help='Dimension of each embedding')

    # loss function
    parser.add_argument('--nu', type=int, default=16384, help='Dimension of the queue')
    parser.add_argument('--tau', type=float, default=1.0, help='temperature')

    # EMA setting
    parser.add_argument('--m', type=float, default=0.999, help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--lr_warmup', type=float, default=10, help='Linear warm-up cycle')
    parser.add_argument('--suffix', type=str, default=None, help='suffix to add on the model name')
    parser.add_argument('--data_folder', type=str, default="./", help='path to data')
    parser.add_argument('--model_path', type=str, default="./model_save", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="./tensorboard", help='path to tensorboard')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_nu{}_{}_lr{}_bsz{}_epoch{}_tau{}'.format(opt.dataset, opt.nu, opt.model,
                                                        opt.learning_rate,opt.batch_size, opt.epochs, opt.tau)
    opt.model_name += opt.suffix

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt

def main():

    args = parse_option()
    if args.gpu is not None:
        print("Use GPU: {} for training".formfat(args.gpu))
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=5)

    #----------------------------------------------prepare dataset--------------------------------------------------------
    image_size, mean, std, n_class = get_dataset_stat(args.dataset)
    args.n_class = n_class
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset, test_dataset = create_dataset(args.dataset, train_transform, train_transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    full_loader = torch.utils.data.DataLoader(full_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=True)
    n_full_data = len(full_dataset)


    #----------------------------------------------prepare model/loss/optimizer--------------------------------------------------------
    model, model_ema = create_model(args.model, args.n_class)
    moment_update(model, model_ema, 0)

    elbo = MiCE_ELBO(args.low_dim, n_full_data, args.nu, tau=args.tau, n_class=args.n_class).cuda(
        args.gpu)

    model = model.cuda()
    model_ema = model_ema.cuda()
    weight_decay = args.weight_decay

    parameters = [{'params': model.parameters(), 'weight_decay': weight_decay},
                  {'params': elbo.parameters(), 'weight_decay': weight_decay}]
    print("Update elbo parameters:", list(parameters[1]['params']), parameters[1]['weight_decay'],)
    optimizer = torch.optim.SGD(parameters,
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=0.0)
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')

            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])

            elbo.load_state_dict(checkpoint['elbo'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for param_group in optimizer.param_groups:
                print(param_group['lr'], param_group['weight_decay'])

            model_ema.load_state_dict(checkpoint['model_ema'])
            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #----------------------------------------------start training-------------------------------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        log_dict = train_MiCE(epoch, full_loader, model, model_ema, elbo, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        for key, val in log_dict.items():
            logger.log_value(key, val, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # saving the model
        if epoch % 5 == 0:
            print('==> Saving to current...')
            state = {
                    'opt': args,
                    'model': model.state_dict(),
                    'elbo': elbo.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
            }
            state['model_ema'] = model_ema.state_dict()
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)
            del state
            torch.cuda.empty_cache()

    return


def train_MiCE(epoch, train_loader, model, model_ema, elbo, optimizer, opt):
    """
    one epoch training for MiCE
    """
    model.train()
    model_ema.eval()
    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    all_cluster_labels = []
    all_true_labels = []
    all_pi_labels = []

    new_center_v = torch.zeros((opt.n_class, opt.low_dim)).cuda()

    end = time.time()
    for idx, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs[0].size(0)
        # ===================forward=====================
        x1, x2, x3 = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)
        f = model(x1)
        with torch.no_grad():
            x2 = x2[shuffle_ids]
            v = model_ema(x2)
            v = v[reverse_ids]

        _, g = model(x3, True)

        loss, out, posterior, pi_logit = elbo(f, v, g)

        # For cluster label
        feat_pseudo_label = torch.argmax(posterior, dim=1)  # batch x 1
        pi_label = torch.argmax(pi_logit, dim=1)
        all_cluster_labels.extend(list(feat_pseudo_label.data.cpu().numpy()))
        all_true_labels.extend(list(target.data.cpu().numpy()))
        all_pi_labels.extend(list(pi_label.data.cpu().numpy()))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        moment_update(model, model_ema, opt.m)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter))
            sys.stdout.flush()

        # Aggregate the teacher's results for the analytical update (Eq.13)
        with torch.no_grad():
            one_hot_pseudo = torch.nn.functional.one_hot(feat_pseudo_label, num_classes=opt.n_class).float()
            new_center_v += torch.einsum("bk,bkd->kd", [one_hot_pseudo, v])

    elbo.update_cluster(new_center_v)

    all_true_labels = np.array(all_true_labels)
    all_cluster_labels = np.array(all_cluster_labels)
    all_pi_labels = np.array(all_pi_labels)
    cluster_acc = acc(all_true_labels, all_cluster_labels)
    pi_cluster_acc = acc(all_true_labels, all_pi_labels)

    log_dict = {}
    log_dict['loss'] = loss_meter.avg

    log_dict['Train_cluster_acc'] = cluster_acc
    log_dict['Train_pi_cluster_acc'] = pi_cluster_acc
    print(np.bincount(all_cluster_labels))
    print("Cluster ACC:", cluster_acc, "PI Cluster ACC:", pi_cluster_acc)

    return log_dict


if __name__ == '__main__':
    main()
