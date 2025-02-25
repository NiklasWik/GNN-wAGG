"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy

"""
    For GCNs
"""

"""  for idx,l in enumerate(model.layers):
           print("------------after loss---------")
            print("iteration: ", iter)
            print("layer: ", idx, ", P: ", l.P)
            print("layer: ", idx, ", grad(P): ", l.P.gra
            print("layer: ", idx, ", A: ", l.A.weight)
            print("layer: ", idx, ", grad A: ", l.A.weight.grad)
            print("layer: ", idx, ", B: ", l.B.weight)
            print("layer: ", idx, ", grad B: ", l.B.weight.grad)
            print("layer: ", idx, ", C: ", l.C.weight)
            print("layer: ", idx, ", .gradC: ", l.C.weight.grad)
            print("layer: ", idx, ", D: ", l.D.weight)
            print("layer: ", idx, ", .gradD: ", l.D.weight.grad)
            print("layer: ", idx, ", E: ", l.E.weight)
            print("layer: ", idx, ", .gradE: ", l.E.weight.grad)
            print("------------loss end---------") """

""" for idx,l in enumerate(model.layers):
            print("------------after backward---------")
            print("iteration: ", iter)
            print("layer: ", idx, ", P: ", l.P)
            print("layer: ", idx, ", grad(P): ", l.P.grad)
            print("layer: ", idx, ", A: ", l.A.weight)
            print("layer: ", idx, ", grad A: ", l.A.weight.grad)
            print("layer: ", idx, ", B: ", l.B.weight)
            print("layer: ", idx, ", grad B: ", l.B.weight.grad)
            print("layer: ", idx, ", C: ", l.C.weight)
            print("layer: ", idx, ", .gradC: ", l.C.weight.grad)
            print("layer: ", idx, ", D: ", l.D.weight)
            print("layer: ", idx, ", .gradD: ", l.D.weight.grad)
            print("layer: ", idx, ", E: ", l.E.weight)
            print("layer: ", idx, ", .gradE: ", l.E.weight.grad) """

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()

        #torch.cuda.empty_cache()

        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc





"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)

        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network_dense(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc
