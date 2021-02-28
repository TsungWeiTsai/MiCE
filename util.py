from __future__ import print_function
import torch
from models.resnet_MiCE import ResNet34
from models.resnet_cifar_MiCE import ResNet34_cifar
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils.linear_assignment_ import linear_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def create_model(model_name, n_label=10):
    if model_name == 'resnet34':
        model = ResNet34(n_label=n_label)
        model_ema = ResNet34(n_label=n_label)
    elif model_name == 'resnet34_cifar':
        model = ResNet34_cifar(n_label=n_label)
        model_ema = ResNet34_cifar(n_label=n_label)
    else:
        raise NotImplementedError('model not supported {}'.format(model_name))

    return model, model_ema


def adjust_learning_rate(epoch, opt, optimizer):
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if epoch <= opt.lr_warmup:  # warm up epoch = 10
        print("Linear warm up")
        decay_weight = linear_rampup(epoch, opt.lr_warmup)

        for param_group in optimizer.param_groups:
            param_group['lr'] =  opt.learning_rate * decay_weight
        return

    if steps > 0:
        decay_weight = (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] =  opt.learning_rate * decay_weight


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def acc(y_true, y_pred):
    """
    https://github.com/XifengGuo/DEC-keras/blob/master/metrics.py

    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


if __name__ == '__main__':
    print()