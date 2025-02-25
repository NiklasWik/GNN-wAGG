import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GATLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
            
        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h # for residual connection

        h = self.gatconv(g, h).flatten(1)
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        return h
    

##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class CustomGATHeadLayer(nn.Module):
    def __init__(self, neighbor_aggr, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        if neighbor_aggr == "pnorm":
            self._reduce = self.reduce_p
            self.p = nn.Parameter(torch.rand(out_dim)*3+1)
        elif neighbor_aggr == "pnorm_robust":
            self._reduce = self.reduce_p_robust
            self.p = nn.Parameter(torch.rand(out_dim)*3+1)
        elif neighbor_aggr == "planar_sig":
            self._reduce = self.reduce_planar_sigmoid
            self.w = nn.Parameter(torch.rand(out_dim)-1/2)
            self.b = nn.Parameter((torch.rand(out_dim)-1/2)-6)
        elif neighbor_aggr == "planar_tanh":
            self._reduce = self.reduce_tanh
            self.w = nn.Parameter(torch.rand(out_dim)-4.5)
            self.b = nn.Parameter((torch.rand(out_dim)*0.01-0.01))
        elif neighbor_aggr == "planar_leaky":
            raise NotImplementedError(' LeakyReLU not implemented in layer.')
            #self._reduce = self.reduce_planar_leaky
        elif neighbor_aggr == "sum":
            self._reduce = self.reduce_func
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))


    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def reduce_planar_sigmoid(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        msg = (alpha * nodes.mailbox['z'])

        w = torch.exp(self.w)
        fsum = torch.sum(torch.sigmoid(w*msg+self.b), dim=1)
        
        sig_in = torch.clamp(fsum/torch.max(fsum), 0.000001, 0.9999999)
        
        """ print("max: ", torch.max(fsum))
        print("min: ", torch.min(fsum)) """

        out_h = (torch.log(sig_in/(1-sig_in))-self.b)/w
        return {'h': out_h}

    def reduce_tanh(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        msg = (alpha * nodes.mailbox['z'])

        w = torch.exp(self.w)
        msg = w * msg + self.b
        
        fsum = torch.clamp(torch.sum(torch.tanh(msg), dim=1), -0.9999999, 0.9999999)
        
        """ print("max: ", torch.max(fsum))
        print("min: ", torch.min(fsum)) """

        out_h = (torch.atanh(fsum) - self.b) / w
        return {'h': out_h}
        
    def reduce_planar_leaky(self, nodes):
        """
        Not Implemented
        sigmoid clone below
        """
        w = torch.exp(self.w)
        msg = torch.abs(nodes.mailbox['e'])
        fsum = torch.sum(torch.sigmoid(w*msg+self.b), dim=1)
        sig_in = torch.clamp(fsum/torch.max(fsum), 0.000001, 0.9999999)
        out_h = (torch.log(sig_in/(1-sig_in))-self.b)/w
        return {'h': out_h}

    def reduce_p(self, nodes):
        p = torch.clamp(self.p,1,100)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        eps = 1e-6
        h = alpha * nodes.mailbox['z']
        h = torch.abs(h + eps).pow(p)
        return {'h': (torch.sum(h, dim=1) + eps).pow(1/p)}
    
    def reduce_p_robust(self, nodes):
        p = torch.clamp(self.p,1,100)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        eps = 1e-6
        h = alpha * nodes.mailbox['z']
        a = torch.max(h)
        h = torch.abs(h/a + eps).pow(p)
        return {'h': (torch.sum(h, dim=1) + eps).pow(1/p)*a}

    def forward(self, g, h, e):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self._reduce)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, neighbor_aggr, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayer(neighbor_aggr, in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h, e) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    
##############################################################


class CustomGATHeadLayerEdgeReprFeat(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc_h = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_e = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_proj = nn.Linear(3* out_dim, out_dim)
        self.attn_fc = nn.Linear(3* out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.batchnorm_e = nn.BatchNorm1d(out_dim)

    def edge_attention(self, edges):
        z = torch.cat([edges.data['z_e'], edges.src['z_h'], edges.dst['z_h']], dim=1)
        e_proj = self.fc_proj(z)
        attn = F.leaky_relu(self.attn_fc(z))
        return {'attn': attn, 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
    
    def forward(self, g, h, e):
        z_h = self.fc_h(h)
        z_e = self.fc_e(e)
        g.ndata['z_h'] = z_h
        g.edata['z_e'] = z_e
        
        g.apply_edges(self.edge_attention)
        
        g.update_all(self.message_func, self.reduce_func)
        
        h = g.ndata['h']
        e = g.edata['e_proj']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)
        
        h = F.elu(h)
        e = F.elu(e)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    

class CustomGATLayerEdgeReprFeat(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerEdgeReprFeat(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        e_in = e

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp)
            head_outs_e.append(e_temp)

        if self.merge == 'cat':
            h = torch.cat(head_outs_h, dim=1)
            e = torch.cat(head_outs_e, dim=1)
        else:
            raise NotImplementedError

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    
##############################################################


class CustomGATHeadLayerIsotropic(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayerIsotropic(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerIsotropic(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
