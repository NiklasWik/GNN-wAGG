import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedTestLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, aggr_type, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if aggr_type == "pnorm":
            self._reducer = self.reduce_p
            self.P = nn.Parameter(torch.rand(output_dim)*3+1)
        if aggr_type == "pnorm_robust":
            self._reducer = self.reduce_p_robust
            self.P = nn.Parameter(torch.rand(output_dim)*3+1)
        elif aggr_type == "planar_sig":
            self._reducer = self.reduce_sig
            self.w = nn.Parameter(torch.rand(output_dim)-1)
            self.b = nn.Parameter((torch.rand(output_dim)*1-6.5))
        elif aggr_type == "planar_tanh":
            self._reducer = self.reduce_tanh
            self.w = nn.Parameter(torch.rand(output_dim)-10)
            self.b = nn.Parameter((torch.rand(output_dim)*0.01-0.01/2))
        elif aggr_type == "planar_relu":
            self._reducer = self.reduce_relu
            self.w = nn.Parameter(torch.rand(output_dim)*1)
            self.b = nn.Parameter((torch.rand(output_dim)*1-3))
            self.R = nn.Parameter((torch.rand(1)*0.1)+0.2)
        elif aggr_type == "sum":
            self._reducer = self.reduce_sum
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))

        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def reduce_sum(self, nodes):
        print(torch.max(nodes.mailbox['m']))
        return {'sum_sigma_h': torch.sum(nodes.mailbox['m'], dim=1)}

    def reduce_sig(self, nodes):
        w = torch.exp(self.w)
        msg = nodes.mailbox['m']
        fsum = torch.sum(torch.sigmoid(w*msg+self.b), dim=1)
        sig_in = torch.clamp(fsum, 0.000001, 0.9999999)
        out_h = (torch.log(sig_in/(1-sig_in))-self.b)/w
        return {'sum_sigma_h': out_h}

    def reduce_relu(self, nodes):
        w = torch.exp(self.w)
        R = torch.clamp(self.R, 0.00001, 0.999999)
        msg = w * nodes.mailbox['m'] + self.b
        print("min: ", torch.min(msg))
        print("max: ", torch.max(msg))
        fsum = torch.sum(torch.maximum(msg, R * msg), dim=1)
        out_h = (torch.minimum(fsum, fsum / R) - self.b) / w
        return {'sum_sigma_h': out_h}

    def reduce_tanh(self, nodes):
        w = torch.exp(self.w)
        msg = w * nodes.mailbox['m'] + self.b

        """ print("MSG max: ", torch.max(msg))
        print("MSG min: ", torch.min(msg)) """
        
        fsum = torch.clamp(torch.sum(torch.tanh(msg), dim=1), -0.99, 0.99)
        
        """ print("max: ", torch.max(fsum))
        print("min: ", torch.min(fsum)) """

        out_h = (torch.atanh(fsum) - self.b) / w
        return {'sum_sigma_h': out_h}

    def reduce_p(self,nodes):
        p = torch.clamp(self.P,1,100)
        h = torch.abs(nodes.mailbox['m'])
        alpha = torch.max(h)
        eps = 1e-4
        h = torch.pow(h/alpha + eps, p)
        
        return {'sum_sigma_h': torch.pow(torch.sum(h, dim=1) + eps , 1/p) * alpha}

    def reduce_p(self,nodes):
        p = torch.clamp(self.P,1,100)
        h = torch.abs(nodes.mailbox['m'])
        eps = 1e-4
        h = torch.pow(h + eps, p)
        
        return {'sum_sigma_h': torch.pow(torch.sum(h, dim=1) + eps , 1/p)}
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma')) 

        g.ndata['eee'] = g.ndata['Bh'] / (g.ndata['sum_sigma'] + 1e-6) ### bring here
        
        g.update_all(fn.u_mul_e('eee', 'sigma', 'm'), self._reducer) 

        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h']  # dis here 

        #h, e = self.update_all_p_norm(g)
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
       
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class GatedGCNLayerEdgeFeatOnly(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        #g.update_all(self.message_func,self.reduce_func) 
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'e'))
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


##############################################################


class GatedGCNLayerIsotropic(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h)
        #g.update_all(self.message_func,self.reduce_func) 
        g.update_all(fn.copy_u('Bh', 'm'), fn.sum('m', 'sum_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_h']
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    
