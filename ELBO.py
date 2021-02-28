import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

class MiCE_ELBO(nn.Module):
    def __init__(self, inputSize, outputSize, nu, tau=1.0, n_class=10):
        super(MiCE_ELBO, self).__init__()

        print("--------------------Initializing -----------------------------------")
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = nu
        self.tau = tau
        self.kappa = self.tau
        self.index = 0
        self.n_class = n_class

        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('queue', torch.rand(self.queueSize, self.n_class, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{},{})'.format(self.queueSize, self.n_class, inputSize))

        if n_class==10:
            cluster_file = np.loadtxt("./kernel_paras/meanvar1_featuredim128_class10.mat")
        elif n_class == 20:
            cluster_file = np.loadtxt("./kernel_paras/meanvar1_featuredim128_class20.mat")
        elif n_class ==15:
            cluster_file = np.loadtxt("./kernel_paras/meanvar1_featuredim128_class15.mat")

        self.register_buffer('gating_prototype', torch.from_numpy(cluster_file).type(torch.FloatTensor))
        self.expert_prototype = nn.Parameter(torch.Tensor(self.n_class, inputSize))

        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        self.logSoftmax2 = torch.nn.LogSoftmax(dim=2)
        print("--------------------Initialization Ends-----------------------------------")

    def update_cluster(self, new_center):

        with torch.no_grad():
            new_center = F.normalize(new_center, dim=1)
            out_ids = torch.arange(self.n_class).cuda()
            out_ids = out_ids.long()  # BS x 1
            self.expert_prototype.index_copy_(0, out_ids, new_center) 

    def forward(self, f, v, g,):
        batchSize = f.shape[0] 
        v = v.detach()

        pi_logit_student = torch.div(torch.einsum('kd,bd->bk', [self.gating_prototype.detach().clone(), g]),
                                         self.kappa)  # K x D  vs B x D ---> BS x K
        log_pi = self.logSoftmax(pi_logit_student + 1e-18)
        pi = torch.exp(log_pi)  # p(z | x)

        # positive sample
        v_f = torch.einsum("bkd,bkd->bk", [f, v]).unsqueeze(-1) # BS x K x D --> BS x K x 1
        v_mu = torch.einsum("bkd,kd->bk", [v, F.normalize(self.expert_prototype, dim=1)]).unsqueeze(-1)  # BS x K x 1

        l_pos = (v_f + v_mu)  # BS x K x 1
        l_pos = torch.einsum('bki->kbi', [l_pos]) # BS x K x 1 --> K x BS x 1

        # Negative sample
        queue = self.queue.detach().clone()  # nu x D x K
        queue_f = torch.einsum('vkd,bkd->kbv', [queue, f])  # K x BS x nu
        queue_mu = torch.einsum('vkd,kd->kv', [queue, F.normalize(self.expert_prototype, dim=1)]).unsqueeze(1)  # K x 1 x nu
        del queue

        l_neg = queue_f + queue_mu  # K x BS x nu
        out = torch.div(torch.cat([l_pos, l_neg], dim=2), self.tau) # K x BS x (nu + 1)
        del l_pos, l_neg

        log_phi = self.logSoftmax2(out + 1e-18)
        normalized_phi = torch.exp(log_phi) # p(y | x, z)

        log_phi_pos = log_phi[:, :, 0]
        normalized_phi_pos = normalized_phi[:, :, 0]
        del normalized_phi

        normalized_phi_pos = normalized_phi_pos.transpose(1, 0) * pi                # BS x K
        posterior = torch.div(normalized_phi_pos, normalized_phi_pos.sum(1).view(-1, 1)).squeeze().contiguous()  # BS x K: posterior -> each row = p(z | v_i, x_i)  # BS x Classes, probability
        log_phi_pos = log_phi_pos.transpose(1, 0).squeeze().contiguous() # K x BS -> BS x K
        loss = -torch.sum(posterior * (log_pi + log_phi_pos - torch.log(posterior + 1e-18))) / float(batchSize)

        # update queue using EMA predictions
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long() 

            self.queue.index_copy_(0, out_ids, v)
            self.index = (self.index + batchSize) % self.queueSize  
        return loss, loss, posterior, log_pi 
