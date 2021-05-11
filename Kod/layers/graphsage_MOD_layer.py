import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import SAGEConv

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False, 
                 bias=True, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)

        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
                                bias=bias)
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        if aggregator_type == "maxpool":
            print("max")
            self._reducer = self.maxreduce
        elif aggregator_type == "pnorm":
            print("pnorm")
            self.power = nn.Parameter(torch.rand(in_feats)*6+1) 
            self._reducer = self.reduce_p
        elif aggregator_type == "planar_sig":
            print("planar_sig, cluster inits")
            self._reducer = self.reduce_planar
            self.w = nn.Parameter(torch.rand(in_feats)-1)
            self.b = nn.Parameter((torch.rand(in_feats)*1-10))
        elif aggregator_type == "planar_tanh":
            print("planar_tanh, cluster inits")
            self._reducer = self.reduce_tanh
            self.w = nn.Parameter(torch.rand(in_feats)-3.5)
            self.b = nn.Parameter((torch.rand(in_feats)*0.01-0.05))
        elif aggregator_type == "sum":
            self._reducer = self.reduce_sum
        else:
            self.aggregator = MeanAggregator()
            print("DU KÖR MED MEAN??? DET HÄR FUNKAR INTE")
            #gotta love it
        
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)

    def maxreduce(self, nodes):
        return {'c': torch.max(nodes.mailbox['m'], dim=1)}
    
    def reduce_sum(self, nodes):
        return {'c': torch.sum(nodes.mailbox['m'], dim=1)}

    def reduce_p(self, nodes):
        #p = torch.clamp(self.aggregator.power,1,100)
        p = torch.clamp(self.power,1,100)
        h = torch.abs(nodes.mailbox['m'])
        alpha = torch.max(h)
        eps = 1e-6
        h = torch.pow(torch.div(h,alpha) + eps ,p)
        return {'c': torch.pow(torch.sum(h, dim=1) + eps ,torch.div(1,p))*alpha}

    def reduce_planar(self, nodes):
        w = torch.exp(self.w)
        msg = nodes.mailbox['m']
        fsum = torch.sum(torch.sigmoid(w*msg+self.b), dim=1)
        #sig_in = torch.clamp(fsum/torch.max(fsum), 0.000001, 0.9999999)
        """ print(torch.max(fsum))
        print(torch.min(fsum)) """
        sig_in = torch.clamp(fsum, 0.000001, 0.9999999)
        out_h = (torch.log(sig_in/(1-sig_in))-self.b)/w
        return {'c': out_h}

    def reduce_tanh(self, nodes):
        w = torch.exp(self.w)
        msg = w * nodes.mailbox['m'] + self.b

        print("MSG max: ", torch.max(msg))
        print("MSG min: ", torch.min(msg))
        
        fsum = torch.clamp(torch.sum(torch.tanh(msg), dim=1), -0.9999999, 0.9999999)
        
        """ print("max: ", torch.max(fsum))
        print("min: ", torch.min(fsum)) """

        out_h = (torch.atanh(fsum) - self.b) / w
        return {'c': out_h}

    def forward(self, g, h):
        h_in = h              # for residual connection
        
        h = self.dropout(h)
        g.ndata['h'] = h
        g.ndata['h'] = self.linear(g.ndata['h'])
        g.ndata['h'] = self.activation(g.ndata['h'])
        
        g.update_all(fn.copy_src('h', 'm'), self._reducer, self.nodeapply)
            
        h = g.ndata['h']

        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.aggregator_type, self.residual)

    

    
class NodeApply(nn.Module):
    """
    Works -> the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}
    
    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################



class GraphSageLayerEdgeFeat(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False, 
                 bias=True, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.activation = activation
        
        self.A = nn.Linear(in_feats, out_feats, bias=bias)
        self.B = nn.Linear(in_feats, out_feats, bias=bias)
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout, bias=bias)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
            
    def message_func(self, edges):
        Ah_j = edges.src['Ah']    
        e_ij = edges.src['Bh'] + edges.dst['Bh'] # e_ij = Bhi + Bhj
        edges.data['e'] = e_ij
        return {'Ah_j' : Ah_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        # Anisotropic MaxPool aggregation
        
        Ah_j = nodes.mailbox['Ah_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        
        Ah_j = sigma_ij * Ah_j
        if self.activation:
            Ah_j = self.activation(Ah_j)
           
        c = torch.max(Ah_j, dim=1)[0]
        return {'c' : c}

    def forward(self, g, h):
        h_in = h              # for residual connection
        h = self.dropout(h)
        
        g.ndata['h']  = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.update_all(self.message_func, 
                     self.reduce_func,
                     self.nodeapply)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.residual)


##############################################################


class GraphSageLayerEdgeReprFeat(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False, 
                 bias=True, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.activation = activation
        
        self.A = nn.Linear(in_feats, out_feats, bias=bias)
        self.B = nn.Linear(in_feats, out_feats, bias=bias)
        self.C = nn.Linear(in_feats, out_feats, bias=bias)
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout, bias=bias)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
            self.batchnorm_e = nn.BatchNorm1d(out_feats)
            
    def message_func(self, edges):
        Ah_j = edges.src['Ah']    
        e_ij = edges.data['Ce'] + edges.src['Bh'] + edges.dst['Bh'] # e_ij = Ce_ij + Bhi + Bhj
        edges.data['e'] = e_ij
        return {'Ah_j' : Ah_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        # Anisotropic MaxPool aggregation
        
        Ah_j = nodes.mailbox['Ah_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        
        Ah_j = sigma_ij * Ah_j
        if self.activation:
            Ah_j = self.activation(Ah_j)
           
        c = torch.max(Ah_j, dim=1)[0]
        return {'c' : c}

    def forward(self, g, h, e):
        h_in = h              # for residual connection
        e_in = e
        h = self.dropout(h)
        
        g.ndata['h']  = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 
        g.update_all(self.message_func, 
                     self.reduce_func,
                     self.nodeapply)
        h = g.ndata['h']
        e = g.edata['e']
        
        if self.activation:
            e = self.activation(e) # non-linear activation
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, 
            self.residual)