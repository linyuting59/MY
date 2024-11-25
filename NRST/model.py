import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
use_cuda = torch.cuda.is_available()


class IO_B(nn.Module):
    """docstring for IO_B"""

    def __init__(self, dim, hid):
        super(IO_B, self).__init__()
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)

    def forward(self, x, adj_col, i, num, node_size):
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        #num =node_num//node_size ,node_num is the total number of nodes
        #node_size: In order to save memory, the information of node i is only
        # combined with the information of node_size nodes at a time
        #eg.We have a total of 2000 node information,
        # and setting node_size to 800 means that the i-th node only ocombined with the information of 800 nodes at a time at a time.
        # At this time, num=2000//800=2
        starter = x[:, i, :]
        x_total_sum = 0
        for n in range(num + 1):
            if n != num:
                current_x = x[:, n * node_size:(n + 1) * node_size, :]
                current_adj_col = adj_col[n * node_size:(n + 1) * node_size]
            else:
                current_x = x[:, n * node_size:, :]
                current_adj_col = adj_col[n * node_size:]
            ender = x[:, i, :]
            ender = ender.unsqueeze(1)
            ender = ender.expand(current_x.size(0), current_x.size(1), current_x.size(2))
            c_x = torch.cat((current_x, ender), 2)

            c_x = F.relu(self.n2e(c_x))
            c_x = F.relu(self.e2e(c_x))

            c_x = c_x * current_adj_col.unsqueeze(1).expand(current_adj_col.size(0), self.hid)
            current_x_sum = torch.sum(c_x, 1)
            x_total_sum = x_total_sum + current_x_sum

        x = F.relu(self.e2n(x_total_sum))
        x = F.relu(self.n2n(x))

        x = torch.cat((starter, x), dim=-1)
        x = self.output(x)
        return x


class IO_B_Voter(nn.Module):
    """docstring for IO_B"""

    def __init__(self, dim, hid):
        super(IO_B_Voter, self).__init__()
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, adj_col, i, num, node_size):
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        #num =node_num//node_size ,node_num is the total number of nodes
        #node_size: In order to save memory, the information of node i is only
        # combined with the information of node_size nodes at a time
        #eg.We have a total of 2000 node information,
        # and setting node_size to 800 means that the i-th node only ocombined with the information of 800 nodes at a time at a time.
        # At this time, num=2000//800=2
        starter = x[:, i, :]
        x_total_sum = 0
        for n in range(num + 1):
            if n != num:
                current_x = x[:, n * node_size:(n + 1) * node_size, :]
                current_adj_col = adj_col[n * node_size:(n + 1) * node_size]
            else:
                current_x = x[:, n * node_size:, :]
                current_adj_col = adj_col[n * node_size:]
            ender = x[:, i, :]
            ender = ender.unsqueeze(1)
            ender = ender.expand(current_x.size(0), current_x.size(1), current_x.size(2))
            c_x = torch.cat((current_x, ender), 2)

            c_x = F.relu(self.n2e(c_x))
            c_x = F.relu(self.e2e(c_x))

            c_x = c_x * current_adj_col.unsqueeze(1).expand(current_adj_col.size(0), self.hid)
            current_x_sum = torch.sum(c_x, 1)
            x_total_sum = x_total_sum + current_x_sum

        x = F.relu(self.e2n(x_total_sum))
        x = F.relu(self.n2n(x))

        x = torch.cat((starter, x), dim=-1)
        x = self.output(x)
        x = self.logsoftmax(x)

        return x

#####################
# Network Generator #
#####################


class Gumbel_Generator_Old(nn.Module):
    def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_Old, self).__init__()
        self.sz = sz
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        self.temperature = self.temperature * self.temp_drop_frac

    def sample_all(self, hard=False,epoch=1):
        device = self.gen_matrix.device
        self.logp = self.gen_matrix.view(-1, 2)
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2, device=device)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
        out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix

    def sample_adj_i(self, i, hard=False, sample_time=1):
        self.logp = self.gen_matrix[:, i]
        out = gumbel_softmax(self.logp, self.temperature, hard=hard)
        out_matrix = out.float() if hard else out[:, 0]
        return out_matrix

    def get_temperature(self):
        return self.temperature

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)

    def init_from_previous(self, prev_gen_matrix):
        with torch.no_grad():
            self.gen_matrix.data = prev_gen_matrix.clone()
    def load_from_adj(self, adj_matrix):
        device = self.gen_matrix.device
        eps = 1e-8
        prob = adj_matrix.to(device) + eps
        prob = prob / (prob.sum(dim=-1, keepdim=True) + eps)
        log_prob = torch.log(prob)
        self.gen_matrix.data = log_prob.unsqueeze(-1).repeat(1, 1, 2)
        self.gen_matrix.data[:, :, 1] = torch.log(1 - prob + eps)

def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    if use_cuda:
        gumbel = gumbel.cuda()
    return gumbel


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=1)



def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = logits.size()[-1]
        y_hard = torch.max(y.data, 1)[1]
        y = y_hard
    return y




