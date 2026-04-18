import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils



DEVICE = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')

class GraphCNNLayer(nn.Module):
    def __init__(self, n_feats, adj_chans=4, n_filters=64, bias=True):
        super(GraphCNNLayer, self).__init__()
        self.n_feats = n_feats
        self.adj_chans = adj_chans
        self.n_filters = n_filters
        self.has_bias = bias

        # [C*L, F], C = n_feats, L = adj_chans, F = n_filters; this is for the edge feats 
        self.weight_e = nn.Parameter(torch.FloatTensor(adj_chans*n_feats, n_filters))
        # [C, F], this is for 𝐈𝐕in𝐖0
        self.weight_i = nn.Parameter(torch.FloatTensor(n_feats, self.n_filters))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_filters))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.weight_i)

        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, A):
        '''V node features: [b, N, C], A adjs: [b, N, L, N], L = adj_chans'''
        b, N, C = V.shape
        b, N, L, _ = A.shape

        # formula: 𝐕out = 𝐈𝐕in𝐖0 + GConv(𝐕in, 𝐹) + 𝐛; 𝐈𝐕in = 𝐕in, so 𝐈𝐕in𝐖0 = 𝐕in𝐖0
        
        # A [b, N, L, N] -> [b, N*L, N]
        A_reshape = A.view(-1, N*L, N)
        # [b, N*L, N] * [b, N, C] -> [b, N*L, C]
        n = torch.bmm(A_reshape, V)
        # [b, N*L, C] -> [b, N, L*C]
        n = n.view(-1, N, L*self.n_feats)

        # n [b, N, L*C], W [C*L, F], V [b, N, C], W_I [C, F]
        # -> [b, N, F] + [b, N, F] + b
        output = torch.matmul(n, self.weight_e) + torch.matmul(V, self.weight_i)

        if self.has_bias:
            output += self.bias

        # output: [b, N, F]
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},adj_chans={self.adj_chans},n_filters={self.n_filters},bias={self.has_bias}) -> [b, N, {self.n_filters}]'

class MultiHeadGlobalAttention(nn.Module):
    '''Input [b, N, C] -> output [b, n_head*C] if concat or else [b, n_head]'''
    def __init__(self, n_feats, n_head=5, alpha=0.2, concat=True, bias=True):
        super(MultiHeadGlobalAttention, self).__init__()

        self.n_feats = n_feats
        self.n_head = n_head
        self.alpha = alpha
        self.concat = concat
        self.has_bias = bias

        self.weight = nn.Parameter(torch.FloatTensor(n_feats, n_head*n_feats))
        self.tune_weight = nn.Parameter(torch.FloatTensor(1, n_head, n_feats))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_head*n_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.tune_weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, graph_size):
        # Gather V of mols in a batch, after this, the pad was removed.
        #print(248, V.shape, graph_size)
        if V.shape[0] == 1:
            Vg = torch.squeeze(V)
            graph_size = [graph_size]
        else:
            Vg = torch.cat([torch.split(v.view(-1, v.shape[-1]), graph_size[i])[0] for i,v in enumerate(torch.split(V, 1))], dim=0)

        Vg = torch.matmul(Vg, self.weight)
        if self.has_bias:
            Vg += self.bias
        Vg = Vg.view(-1, self.n_head, self.n_feats)

        alpha = torch.mul(self.tune_weight, Vg)
        alpha = torch.sum(alpha, dim=-1)
        alpha = F.leaky_relu(alpha, self.alpha) # original code is "alpha = tf.nn.leaky_relu(alpha, alpha=0.2)"
        alpha = utils.segment_softmax(alpha, graph_size)

        #alpha_collect = torch.mean(alpha, dim=-1) # origin code like this. alpha_collect not used?
        alpha = alpha.view(-1, self.n_head, 1)
        V = torch.mul(Vg, alpha)

        if self.concat:
            V = utils.segment_sum(V, graph_size)
            V = V.view(-1, self.n_head*self.n_feats)
        else:
            V = torch.mean(V, dim=1)
            V = utils.segment_sum(V, graph_size)

        return V

    def __repr__(self):
        if self.concat:
            outc = self.n_head*self.n_feats
        else:
            outc = self.n_head
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_head={self.n_head},alpha={self.alpha},concat={self.concat},bias={self.has_bias}) -> [b, {outc}]'

class GConvBlockNoGF(nn.Module):
    def __init__(   self,
                    n_feats,
                    n_filters,
                    mols=1,
                    adj_chans=4,
                    bias=True):

        super(GConvBlockNoGF, self).__init__()

        self.n_feats = n_feats
        self.n_filters = n_filters
        self.mols = mols
        self.adj_chans = adj_chans
        self.has_bias = bias

        #self.graph_conv = GraphCNNLayer(n_feats+n_filters, adj_chans, n_filters, bias)
        self.graph_conv = GraphCNNLayer(n_feats, adj_chans, n_filters, bias)

        #self.bn_global = nn.BatchNorm1d(n_filters*mols)
        self.bn_graph  = nn.BatchNorm1d(n_filters)

    def forward(self, V, A):
        ######## Graph Convolution #########
        # V shape from [b, N, C+F] to [b, N, F1], F1 is n_filters
        V = self.graph_conv(V, A)
        V = self.bn_graph(V.transpose(1, 2).contiguous())
        V = F.relu(V.transpose(1, 2))

        return V

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_filters={self.n_filters},mols={self.mols},adj_chans={self.adj_chans},bias={self.has_bias}) -> [b, N, {self.n_filters}]' 


class MultiHeadMSAttentionPool(nn.Module):
    '''Input [B, L, C] -> output [B, n_head*C] if concat else [B, n_head]'''
    def __init__(self, n_feats, n_head=5, alpha=0.2, concat=True, bias=True):
        super().__init__()
        self.n_feats = n_feats
        self.n_head = n_head
        self.alpha = alpha
        self.concat = concat
        self.has_bias = bias

        self.weight = nn.Parameter(torch.FloatTensor(n_feats, n_head * n_feats))
        self.tune_weight = nn.Parameter(torch.FloatTensor(1, n_head, n_feats))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_head * n_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.tune_weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, mask):
        """
        V: [B, L, C] — 序列隐藏向量
        mask: [B, L] — True 表示 padding，需要被忽略
        """
        B, L, C = V.shape

        # 线性变换
        Vg = torch.matmul(V, self.weight)  # [B, L, n_head*C]
        if self.has_bias:
            Vg += self.bias
        Vg = Vg.view(B, L, self.n_head, self.n_feats)  # [B, L, n_head, C]

        # 打分
        alpha = (self.tune_weight * Vg).sum(dim=-1)  # [B, L, n_head]
        alpha = F.leaky_relu(alpha, self.alpha)

        # mask 掉 padding 位置
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.n_head)  # [B, L, n_head]
        alpha = alpha.masked_fill(mask_expanded, float('-inf'))

        # softmax 按 L 维度
        alpha = F.softmax(alpha, dim=1)  # [B, L, n_head]
        alpha = alpha.unsqueeze(-1)  # [B, L, n_head, 1]

        # 加权求和
        V_att = Vg * alpha  # [B, L, n_head, C]
        if self.concat:
            V_out = V_att.sum(dim=1).reshape(B, self.n_head * self.n_feats)
        else:
            V_out = V_att.mean(dim=-1).sum(dim=1)  # [B, n_head]

        return V_out

    def __repr__(self):
        if self.concat:
            outc = self.n_head * self.n_feats
        else:
            outc = self.n_head
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_head={self.n_head},alpha={self.alpha},concat={self.concat},bias={self.has_bias}) -> [B, {outc}]'

