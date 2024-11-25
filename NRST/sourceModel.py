
from typing import List
import pandas as pd
from sourceArgs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import scipy.sparse as sp
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


features_path = '../sourceDataset/' + args.network +'_'+args.model+ '_features.pickle'
with open(features_path, 'rb') as f:
    h = pickle.load(f)


class InverseModelGat(nn.Module):
    def __init__(self, vae_model: nn.Module, gnn_model: nn.Module, propagate: nn.Module):
        super(InverseModelGat, self).__init__()

        self.vae_model = vae_model
        self.gnn_model = gnn_model
        self.propagate = propagate

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))

    def forward(self, input_pair, seed_vec):
        device = next(self.gnn_model.parameters()).device
        seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() == 1)).to(device)

        seed_hat, mean, log_var = self.vae_model(input_pair)
        # h = construct_features(adj)
        predictions = self.gnn_model(h, seed_hat)
        predictions = self.propagate(predictions, seed_idx)

        return seed_hat, mean, log_var, predictions

    def loss(self, x, x_hat, mean, log_var, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        # reproduction_loss = F.mse_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = forward_loss + reproduction_loss + 1e-3 * KLD
        return KLD, reproduction_loss, forward_loss, total_loss


class ForwardModelGat(nn.Module):
    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        super(ForwardModelGat, self).__init__()
        self.gnn_model = gnn_model
        self.propagate = propagate
        self.relu = nn.ReLU(inplace=True)

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))

    def forward(self, seed_vec):
        device = next(self.gnn_model.parameters()).device
        seed_vec = seed_vec.to(device)
        # seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() == 1)).to(device)
        seed_idx = (seed_vec == 1).nonzero(as_tuple=False)
        # h = construct_features(adj)
        predictions = self.gnn_model(h, seed_vec)
        predictions = self.propagate(predictions, seed_idx)

        # predictions = (predictions + seed_vec)/2

        predictions = self.relu(predictions)

        return predictions

    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.bn = nn.BatchNorm1d(num_features=latent_dim)

    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        h_ = F.relu(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        # self.prelu = nn.PReLU()

    def forward(self, x):
        h = F.relu(self.FC_input(x))
        h = F.relu(self.FC_hidden_1(h))
        h = F.relu(self.FC_hidden_2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder,beta):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.beta = beta

    def reparameterization(self, mean, var):
        std = torch.exp(0.5 * var)  # standard deviation
        epsilon = torch.randn_like(var)
        return mean + std * epsilon

    def forward(self, x, adj=None):
        if adj != None:
            mean, log_var = self.Encoder(x, adj)
        else:
            mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, log_var)  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


class DiffusionPropagate(nn.Module):
    def __init__(self, prob_matrix, niter):
        super(DiffusionPropagate, self).__init__()

        self.niter = niter

        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))

    def forward(self, preds, seed_idx):
        # import ipdb; ipdb.set_trace()
        # prop_preds = torch.ones((preds.shape[0], preds.shape[1])).to(device)
        device = preds.device

        for i in range(preds.shape[0]):
            prop_pred = preds[i]
            for j in range(self.niter):
                P2 = self.prob_matrix.T * prop_pred.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                prop_pred = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(P3, dim=1)
                # prop_pred[seed_idx[seed_idx[:,0] == i][:, 1]] = 1
                prop_pred = prop_pred.unsqueeze(0)
            if i == 0:
                prop_preds = prop_pred
            else:
                prop_preds = torch.cat((prop_preds, prop_pred), 0)

        return prop_preds


class GNNModelGat(nn.Module):
    def __init__(self, input_dim, hiddenunits: List[int], num_classes, prob_matrix, adj_matrix, bias=True,
                 drop_prob=0.5):
        super(GNNModelGat, self).__init__()

        self.input_dim = input_dim
        self.prob_matrix = nn.Parameter(torch.FloatTensor(prob_matrix), requires_grad=False)
        self.adj_matrix = nn.Parameter(torch.FloatTensor(adj_matrix), requires_grad=False)

        # 创建可变数量的GAT层
        self.gat_layers = nn.ModuleList()
        in_features = 4
        for hidden_dim in hiddenunits:
            self.gat_layers.append(GraphAttentionLayer(in_features, hidden_dim, dropout=drop_prob, alpha=0.2))
            in_features = hidden_dim

        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        self.act_fn = nn.ReLU()

    def forward(self, h, seed_vec):
        h = h.to(device)
        seed_vec = seed_vec.to(device)
        adj_matrix = self.adj_matrix.to(device)
        prob_matrix = self.prob_matrix.to(device)

        # 应用所有GAT层
        attentions = []
        for gat_layer in self.gat_layers:
            h, attention = gat_layer(h, adj_matrix)
            h = self.dropout(h)
            attentions.append(attention)

        # 使用最后一层的注意力权重来调整传播概率
        adjusted_prob_matrix = prob_matrix * attentions[-1]

        for i in range(self.input_dim - 1):
            if i == 0:
                mat = adjusted_prob_matrix.T @ seed_vec.T
                attr_mat = torch.cat((seed_vec.T.unsqueeze(0), mat.unsqueeze(0)), 0)
            else:
                mat = adjusted_prob_matrix.T @ attr_mat[-1]
                attr_mat = torch.cat((attr_mat, mat.unsqueeze(0)), 0)

        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_mat.T)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = torch.sigmoid(self.fcs[-1](self.dropout(layer_inner)))
        return res

    def loss(self, y, y_hat):
        return F.mse_loss(y_hat, y)