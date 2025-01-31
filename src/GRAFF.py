import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree, homophily, to_undirected, is_undirected
from torch_geometric.utils import negative_sampling

from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from typing import Optional

# This code is partially inspired from the GRAFF implementation available at https://github.com/realfolkcode/GRAFF

class Symmetric(torch.nn.Module):
    def forward(self, w):
        # This class implements the method to define the symmetry in the squared matrices.
        return w.triu(0) + w.triu(1).transpose(-1, -2)
    
class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        # The weights are initialized to be non-squared, with 2 additional columns. We cut from two of these
        # two vectors q and r, and then we compute w_diag as described in the paper.
        # This procedure is done in order to easily distribute the mass in its spectrum through the values of q and r
        W0 = W[:, :-2].triu(1)

        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r)

        return W0 + w_diag

class External_W(nn.Module):
    def __init__(self, input_dim, device = 'cpu'):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((1, input_dim)))
        self.reset_parameters()
        self.to(device)
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w)

    def forward(self, x):
        # x * self.w behave like a diagonal matrix op., we multiply each row of x by the element-wise w
        return x * self.w


class Source_b(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.empty(1))
     
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta)
    


    def forward(self, x):
        return x * self.beta


class PairwiseInteraction_w(nn.Module):
    def __init__(self, input_dim, symmetry_type='1', device = 'cpu'):
        super().__init__()
        self.W = torch.nn.Linear(input_dim + 2, input_dim, bias = False)

        if symmetry_type == '1':
            symmetry = PairwiseParametrization()
        elif symmetry_type == '2':
            symmetry = Symmetric()

        parametrize.register_parametrization(
            self.W, 'weight', symmetry, unsafe=True)
        self.reset_parameters()
        self.to(device)
        
    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x):
        return self.W(x)


class MLP(nn.Module):
    def __init__(self, input_features, hidden_dim, n_layers=1, device='cpu', dropout_prob=0.5):
        super(MLP, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_features, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_prob))

        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, data, save_embs = False):
        x = data.x
        encodings = self.mlp(x)

        if save_embs:
            return encodings, [x, encodings]
        
        return encodings
        

class GRAFFConv(MessagePassing):
    def __init__(self, external_w, source_b, pairwise_w, self_loops=True):
        super().__init__(aggr='add')

        self.self_loops = self_loops
        self.external_w = external_w #External_W(self.in_dim, device=device)
        self.beta = source_b #Source_b(device = device)
        self.pairwise_W = pairwise_w #PairwiseInteraction_w(self.in_dim, symmetry_type=symmetry_type, device=device)
   

    def forward(self, x, edge_index, x0):

        # We set the source term, which corrensponds with the initial conditions of our system.

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        out_p = self.pairwise_W(x)

        out = self.propagate(edge_index, x=out_p)

        out = out - self.external_w(x) - self.beta(x0)

        return out

    def message(self, x_j, edge_index, x):
        # Does we need the degree of the row or from the columns?
        # x_i are the columns indices, whereas x_j are the row indices
        row, col = edge_index

        # Degree is specified by the row (outgoing edges)
        deg_matrix = degree(col, num_nodes=x.shape[0], dtype=x.dtype)
        deg_inv = deg_matrix.pow(-0.5)
        
        deg_inv[deg_inv == float('inf')] = 0

        denom_degree = deg_inv[row]*deg_inv[col]

        # Each row of denom_degree multiplies (element-wise) the rows of x_j
        return denom_degree.unsqueeze(-1) * x_j



class LinkPredictor_(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers = 0, bias = False, dropout= 0, device = 'cpu'):
        super().__init__()
        
        self.num_layers = num_layers
        layers = []
        if self.num_layers != 0:
            
            layers.append(nn.Linear(input_dim, output_dim, bias = bias))
            for layer in range(self.num_layers):
                layers.append(nn.Linear(output_dim, output_dim, bias = bias))
        
            layers.append(nn.Linear(output_dim, 1, bias = bias))    
        else:
            layers.append(nn.Linear(input_dim, 1, bias = bias))    

    
        self.layers = nn.Sequential(*layers)
        self.dropout = dropout
        self.to(device)
        self.reset_parameters()
             
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, out, training = False):
        
        if self.num_layers != 0:
            for layer_idx in range(len(self.layers)-1):
                out = self.layers[layer_idx](out)
                out = F.relu(out)
                out = F.dropout(out, p = self.dropout, training = training)
        
        out = self.layers[-1](out)

        out = torch.sigmoid(out)

        return out
    
class GraphClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim, gnn = True, pooling='mean', iteration = False, prev_step = False):
        super(GraphClassifier, self).__init__()
        
        self.gnn = gnn
        self.prev_step = prev_step
        self.iteration = iteration        
        
        
        if self.gnn:

            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        if self.iteration:
            if gnn:
                hidden_dim += 1
            else:
                num_features += 1
                
        if self.prev_step:
            if gnn:
                hidden_dim += 1
            else:
                num_features += 1

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim) if self.gnn else nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


        self.pooling = pooling

        if self.pooling == 'vpa':
            self.vpa = VariancePreservingAggregation()

    def forward(self, x, edge_index, iteration, p_step):
        
        

        if self.gnn:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
        elif self.pooling == 'max':
            x = global_max_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
        elif self.pooling == 'sum':
            x = global_add_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
        elif self.pooling == 'vpa':
            x = self.vpa(x)
        else:
            raise ValueError("Invalid pooling function specified.")
        


        if self.iteration:
            x = torch.cat([x, iteration], dim = -1)
        if self.prev_step:
            x = torch.cat([x, p_step], dim = -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)


class AdaGRAFF(nn.Module):
    def __init__(self, input_feat, hidden_dim, num_layers, input_dropout, normalize = False, step=0.1, symmetry_type='1', delta = True, gnn = True, pooling='mean', iteration = False, prev_step = False, threshold = 0.1, self_loops=False, device='cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(
            input_feat, hidden_dim, bias=False)

        self.external_w = External_W(hidden_dim, device=device)
        self.source_b = Source_b(device=device)
        self.pairwise_w = PairwiseInteraction_w(
            hidden_dim, symmetry_type=symmetry_type, device=device)

        self.GRAFF = GRAFFConv(self.external_w, self.source_b, self.pairwise_w,
                                 self_loops=self_loops)
        self.GC = GraphClassifier(hidden_dim, hidden_dim, gnn = gnn, pooling=pooling, iteration = iteration, prev_step = prev_step)
        self.num_layers = num_layers
        
        self.normalize = normalize
        self.delta = delta
        self.iteration = iteration
        self.prev_step = prev_step
        self.threshold = threshold


        if self.normalize:
            self.batch1 = torch.nn.BatchNorm1d(hidden_dim)

        
        self.step = step
        self.drop = torch.nn.Dropout(input_dropout)
        self.reset_parameters()
        self.to(device)
        self.dev = device
        self.max_depth = self.num_layers

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.external_w.reset_parameters()
        self.source_b.reset_parameters()
        self.pairwise_w.reset_parameters()



    def forward(self, data, eval_energy = False):

        x, edge_index = data.x.clone(), data.edge_index.clone()
        
        x = self.enc(x)

        if self.normalize:
            x = self.batch1(x)

        x = enc_out = self.drop(x)

        x0 = enc_out.clone()
        
        if eval_energy:
            with torch.no_grad():
                e_list = []
                energy = Dirichlet_param(x, edge_index, self, x0)
                e_list.append(energy)

        class_list = []

        for i in range(self.num_layers):
            
            if i == 0:
    
                delta_x = F.relu(self.GRAFF(x, edge_index, x0))
                x = x + self.step*delta_x

                p_step = torch.tensor([[self.step]], device = self.dev) if self.prev_step else None
            
            else:
                # print('ciao1')
                delta_x = F.relu(self.GRAFF(x, edge_index, x0))
                it = torch.tensor([[i]], device = self.dev) if self.iteration else None
                # print(delta_x.clone().detach())
                # print(it)
                # print(p_step)

                if self.delta:
                    classification = self.GC((delta_x.clone()).detach(), edge_index, iteration = it, p_step = p_step)
                    
                else:
                    classification = self.GC((x.clone()).detach(), edge_index, iteration = it, p_step = p_step)
                

                if classification < self.threshold:
                    self.max_depth = i+1
                    print("The situation is: ", classification, i+1)

                    return x
                
                x = x + classification*delta_x

                # print("Ciao2")
                
                p_step = torch.tensor([[classification.clone().detach()]], device = self.dev) if self.prev_step else None

                class_list.append(classification.item())

            if eval_energy:
                with torch.no_grad():
                    energy = Dirichlet_param(x, edge_index, self, x0)
                    e_list.append(energy)

        if eval_energy:
            return x, e_list
        else: 
            return x

class PhysicsGNN_LP(nn.Module):
    def __init__(self, input_feat, hidden_dim, num_layers, input_dropout, normalize = False, step=0.1, symmetry_type='1', self_loops=False, device='cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(
            input_feat, hidden_dim, bias=False)

        self.external_w = External_W(hidden_dim, device=device)
        self.source_b = Source_b(device=device)
        self.pairwise_w = PairwiseInteraction_w(
            hidden_dim, symmetry_type=symmetry_type, device=device)

        self.GRAFF = GRAFFConv(self.external_w, self.source_b, self.pairwise_w,
                                 self_loops=self_loops)
        
        self.num_layers = num_layers
        
        self.normalize = normalize
        if self.normalize:
            self.batch1 = torch.nn.BatchNorm1d(hidden_dim)

        
        self.step = step
        self.drop = torch.nn.Dropout(input_dropout)
        self.max_depth = self.num_layers

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.external_w.reset_parameters()
        self.source_b.reset_parameters()
        self.pairwise_w.reset_parameters()


    def forward(self, data, eval_energy = False, save_embs = False):

        x, edge_index = data.x.clone(), data.edge_index.clone()
        
        x = self.enc(x)

        if self.normalize:
            x = self.batch1(x)

        if save_embs: 
            list_embs = []
            list_embs.append(x.clone())

        x = enc_out = self.drop(x)

        x0 = enc_out.clone()
        
        if eval_energy:
            with torch.no_grad():
                e_list = []
                energy = Dirichlet_param(x, edge_index, self, x0)
                e_list.append(energy)


        for i in range(self.num_layers):

            x = x + self.step*F.relu(self.GRAFF(x, edge_index, x0))
            #x = x + self.step*(self.GRAFF(x, edge_index, x0))


            if eval_energy:
                with torch.no_grad():
                    energy = Dirichlet_param(x, edge_index, self, x0)
                    e_list.append(energy)
            
            if save_embs:
                list_embs.append(x.clone())

        if eval_energy:
            return x, e_list

        if save_embs:
            return x, list_embs
        else: 
            return x
    
class GNN_LP(nn.Module):
    def __init__(self, input_feat, hidden_dim, num_layers, GNN_args = None, self_loops = True, device='cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(
            input_feat, hidden_dim, bias=False)

        # SAGE    
        # GNN_args = [GNN, GNN_type, aggregation]
        # GAT    
        # GNN_args = [GNN, GNN_type, n_heads]

        self.GNN_args = GNN_args
        GNN = GNN_args[0] # layer of GNN
        GNN_type = GNN_args[1] # GNN name
        self.res = GNN_args[-1] # residual connection


        layers = []

        for i in range(num_layers):

            if GNN_type == "SAGE":
                layers.append(GNN(hidden_dim, hidden_dim, normalize = True, aggr = GNN_args[2]))
            elif GNN_type == "GAT":
                if GNN_args[2] != 1:
                    if len(layers) == 0 and num_layers == 1:
                        layers.append(GNN(hidden_dim, hidden_dim, add_self_loops = self_loops, heads = 1))
                    elif len(layers) == 0:
                        layers.append(GNN(hidden_dim, hidden_dim, add_self_loops = self_loops, heads = GNN_args[2]))
                    elif i == num_layers-1:
                        layers.append(GNN(GNN_args[2]*hidden_dim, hidden_dim, add_self_loops = self_loops, heads = 1))
                    else:
                        layers.append(GNN(GNN_args[2]*hidden_dim, hidden_dim, add_self_loops = self_loops, heads = GNN_args[2]))
                else:
                    layers.append(GNN(hidden_dim, hidden_dim, add_self_loops = self_loops, heads = 1))
    
            elif GNN_type == "GCN":
                layers.append(GNN(hidden_dim, hidden_dim, add_self_loops = self_loops, normalize = True))
        
    
        
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.enc.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, data, save_embs = False):

        x, edge_index = data.x.clone(), data.edge_index.clone()
        

        x = self.enc(x)

        if save_embs: 
            list_embs = []
            list_embs.append(x.clone())
        
        if self.res:
            for layer in self.layers:
                
                if self.GNN_args[1] == 'GAT':
                    if self.GNN_args[2] > 1: 
                        x = F.relu(layer(x, edge_index))
                        
                    elif self.GNN_args[2] == 1:  
                        x = x + F.relu(layer(x, edge_index))
                else:
                    x = x + F.relu(layer(x, edge_index))
                
                if save_embs:
                    list_embs.append(x.clone())
        else:
            for layer in self.layers:
                
                x = F.relu(layer(x, edge_index))
                
                if save_embs:
                    list_embs.append(x.clone())

        if save_embs:
            return x, list_embs
        else:
            return x     

def Dirichlet_param(x, A, model, x_0):
    

    ext = model.external_w.w.T.squeeze(-1)
    enc = x
    enc_0 = x_0

    source = model.source_b.beta
    pairw_enc = model.pairwise_w.W(enc)

    E_ext = sum([torch.dot(enc[i].T, enc[i].T * ext) for i in range(x.shape[0])])

    E_pairwise = sum([torch.dot(enc[A[0,i].item()].T, pairw_enc[A[1,i].item()]) for i in range(A.shape[1])])

    E_source = 2*sum([torch.dot(enc[i].T, enc_0[i].T * source) for i in range(x.shape[0])])

    tot_energy = E_ext - E_pairwise + E_source

    return tot_energy

def get_readout_input(features, edges, mp_edges, readout_type='hadamard', test = False, grad = False):
    
    edge_indices = edges.clone()
    
    if readout_type == 'hadamard':
        
        readout_input = features[edge_indices[0]] * features[edge_indices[1]]
        
    elif readout_type == 'gradient':
        # features.shape does not change with the data split, since the nodes stay the same within the transductive setting
        
        # We use the degree of the message-passing graph to normalize the features
        #assert is_undirected(mp_edges)
        

        degrees = degree(mp_edges[0], num_nodes=features.shape[0], dtype=torch.float)

        # degrees = torch.where(degrees == 0, torch.tensor(1), degrees)

        feature_source = features[edge_indices[0]] / (torch.sqrt(degrees[edge_indices[0]].view(-1, 1)) + 1)
        feature_target = features[edge_indices[1]] / (torch.sqrt(degrees[edge_indices[1]].view(-1, 1)) + 1)
        # print(feature_source)
        diff = feature_target - feature_source
        
        readout_input  = diff*diff # Absolute value is used to ensure that the undirected interpretation is maintained
       
        if grad:

            diff = torch.sum(readout_input, dim = -1)


            return readout_input, torch.sum(diff)
    
    return readout_input 

class VariancePreservingAggregation(Aggregation):

    '''
    A variance-preserving aggregation function.
    '''

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -1,
    ) -> Tensor:

        sum_aggregation = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        counts = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce="sum")

        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))