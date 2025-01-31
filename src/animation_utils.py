import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn
from sklearn import metrics
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree, homophily
from torch_geometric.utils import negative_sampling

from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import os



def compute_auroc(positives_tensor, negatives_tensor):
    # Combine the positive and negative tensors
    all_tensor = torch.cat([positives_tensor, negatives_tensor])

    # Create the corresponding labels (1 for positives, 0 for negatives)
    labels = torch.cat([torch.ones_like(positives_tensor), torch.zeros_like(negatives_tensor)])

    # Calculate the AUROC using sklearn's roc_auc_score function
    auroc = metrics.roc_auc_score(labels.detach().cpu().numpy(), all_tensor.detach().cpu().numpy())

    return auroc
def get_gradients(features, mp_index, edge_indices):
    '''
    Returns the gradients of the edges. 
    Parameters:
    - features: Tensor
        The node features
    - mp_index: Tensor  
        The message passing index
    - edge_indices: Tuple   
        The edge indices
    Returns:
    - gradients: Tensor
        The gradients of the edges
    '''

    degrees = degree(mp_index[0], num_nodes=features.shape[0], dtype=torch.float)

    # degrees = torch.where(degrees == 0, torch.tensor(1), degrees)

    feature_source = features[edge_indices[0]] / (torch.sqrt(degrees[edge_indices[0]].view(-1, 1)) + 1)
    feature_target = features[edge_indices[1]] / (torch.sqrt(degrees[edge_indices[1]].view(-1, 1)) + 1)
    # print(feature_source)
    diff = feature_target - feature_source

    readout_input  = diff*diff # Absolute value is used to ensure that the undirected interpretation is maintained

    gradients = torch.sum(readout_input, dim = -1)

    return gradients

def get_sub_aurocs(labels, pos_edges_vec, neg_edges_vec, pos_edges, neg_edges):
    

    assert pos_edges_vec.shape == neg_edges_vec.shape
    assert pos_edges.shape[-1] == neg_edges.shape[-1]


    homo_edges_pos = (labels[pos_edges[0]] == labels[pos_edges[1]]).int().unsqueeze(1)
    homo_edges_neg = (labels[neg_edges[0]] == labels[neg_edges[1]]).int().unsqueeze(1)
    

    pos_vec_homo = pos_edges_vec[homo_edges_pos.squeeze() == 1]
    neg_vec_homo = neg_edges_vec[homo_edges_neg.squeeze() == 1]

    pos_vec_hetero = pos_edges_vec[homo_edges_pos.squeeze() == 0]
    neg_vec_hetero = neg_edges_vec[homo_edges_neg.squeeze() == 0]


    print("POS. HOMO are: ", pos_vec_homo.shape)
    print("POS. HETERO are: ", pos_vec_hetero.shape)
    print("NEG. HOMO are: ", neg_vec_homo.shape)
    print("NEG. HETERO are: ", neg_vec_hetero.shape)


    auroc_homo = compute_auroc(pos_vec_homo, neg_vec_homo)
    auroc_hetero = compute_auroc(pos_vec_hetero, neg_vec_hetero)
    auroc_mix_hard = compute_auroc(pos_vec_hetero, neg_vec_homo)
    auroc_mix_easy = compute_auroc(pos_vec_homo, neg_vec_hetero)

    return auroc_homo, auroc_hetero, auroc_mix_hard, auroc_mix_easy




def get_edge_vectors(features, labels, mp_index, pos_edges, neg_edges):    
    '''
    Returns two matrices with the following structure:
    [gradient, homophily, label]
    The number of rows corresponds to the number of edges for both positives or negatives edges. While the first column element represent the edge gradient, the 
    second column element tells whether the edge is homophilic or not. The third column element tells whether the edge is positive or negative.
    Parameters:
    - features: Tensor
        The node features
    - labels: Tensor    
        The node labels
    - mp_index: Tensor
        The message passing index
    - pos_edges: Tuple
        The positive index
    - neg_edges: Tuple
        The negative index
    Returns:
    - pos_vectors: Tensor
        The positive edge vectors
    - neg_vectors: Tensor
        The negative edge vectors
    '''
    assert pos_edges.shape[-1] == neg_edges.shape[-1]

    homo_edges_pos = (labels[pos_edges[0]] == labels[pos_edges[1]]).int().unsqueeze(1)
    homo_edges_neg = (labels[neg_edges[0]] == labels[neg_edges[1]]).int().unsqueeze(1)
    pos_gradients = get_gradients(features, mp_index, pos_edges).unsqueeze(1)
    neg_gradients = get_gradients(features, mp_index, neg_edges).unsqueeze(1)

    pos_vectors = torch.cat([pos_gradients, homo_edges_pos, torch.ones(pos_gradients.shape)], dim = -1)
    neg_vectors = torch.cat([neg_gradients, homo_edges_neg, torch.zeros(neg_gradients.shape)], dim = -1)


    pos_vectors_homo = pos_gradients[homo_edges_pos.squeeze() == 1]
    neg_vectors_homo = neg_gradients[homo_edges_neg.squeeze() == 1]


    pos_vectors_hetero = pos_gradients[homo_edges_pos.squeeze() == 0]
    neg_vectors_hetero = neg_gradients[homo_edges_neg.squeeze() == 0]
   
    auroc_homo = compute_auroc(neg_vectors_homo, pos_vectors_homo)
    auroc_hetero = compute_auroc(neg_vectors_hetero, pos_vectors_hetero)
    auroc_mix_hard = compute_auroc(neg_vectors_homo, pos_vectors_hetero)
    auroc_mix_easy = compute_auroc(neg_vectors_hetero, pos_vectors_homo)

    auroc_tot = compute_auroc(neg_gradients, pos_gradients)


    return pos_vectors, neg_vectors, auroc_homo, auroc_hetero, auroc_mix_hard, auroc_mix_easy, auroc_tot, (pos_gradients, neg_gradients)

    

def animate_trajectories(vec_list_pos, vec_list_neg, auroc_homo_list, auroc_hetero_list, max_elem, display=False, model_name='model', dataset_name='data'):
    fig, ax = plt.subplots(figsize=(10.5,10))

    pos = ax.scatter(vec_list_pos[0][:max_elem, 0], vec_list_pos[0][:max_elem, 1], c="b", label=f'Positive Edges')
    neg = ax.scatter(vec_list_neg[0][:max_elem, 0], vec_list_neg[0][:max_elem, 1], c="r", label=f'Negative Edges')

    ax.set_ylabel('Homophily Edge: 1/0', fontsize=34)  
    ax.set_xlabel('$||(∇\mathbf{H}^t)_{i,j}||^2$', fontsize=38)  
    ax.legend(fontsize=28) 
    
    ax.tick_params(axis='both', which='major', labelsize=22)  

    if display:
        directory_name = f'screenshots/{model_name}/{dataset_name}'
        os.makedirs(directory_name, exist_ok=True)

    def update(frame):
        x_pos = vec_list_pos[frame][:max_elem, 0]
        y_pos = vec_list_pos[frame][:max_elem, 1]

        data = np.stack([x_pos, y_pos]).T
        pos.set_offsets(data)

        x_neg = vec_list_neg[frame][:max_elem, 0]
        y_neg = vec_list_neg[frame][:max_elem, 1]

        auroc_homo = auroc_homo_list[frame]
        auroc_hetero = auroc_hetero_list[frame]
        auroc_mean = (auroc_homo + auroc_hetero) / 2
        
        data = np.stack([x_neg, y_neg]).T
        neg.set_offsets(data)
        tot_tensor = torch.cat([x_pos, x_neg])

        ax.set_xlim(left=0, right=torch.max(tot_tensor).item())

        ax.set_title(f'Layer: {frame}/{len(vec_list_pos)-1}; GS({frame}) 0: {np.round(auroc_hetero, 2)}, GS({frame}) 1: {np.round(auroc_homo, 2)}', fontsize=34)

        return (pos, neg)

    #ani = animation.FuncAnimation(fig=fig, func=update, frames=len(vec_list_pos), interval=2000)
    print("DISPLAY IS: ", display)
    
    if display:
        for frame in range(len(vec_list_pos)):
            update(frame)
            plt.savefig(f'{directory_name}/{frame}.pdf')
    else:
        pos_vectors = vec_list_pos[-1][:max_elem, :]
        neg_vectors = vec_list_neg[-1][:max_elem, :]
        plt.scatter(pos_vectors[:, 0], pos_vectors[:, 1], c='blue')
        plt.scatter(neg_vectors[:, 0], neg_vectors[:, 1], c='red')

        plt.xlabel('Squared norm of the edge Gradients', fontsize=18)
        plt.ylabel('1/0 Homophily Edge', fontsize=18)
        plt.title('Edge Gradients after the GNNs', fontsize=20)

    return plt



if __name__ == '__main__':

    data = torch.load('../data/Texas/test_data_5.pt')
    labels = data.y
    pos_labels = data.pos_edge_label_index
    neg_labels = data.neg_edge_label_index[:, :pos_labels[0].shape[-1]]
    features = data.x
    edge_index = data.edge_index


    l_pos = []
    l_neg = []
    l_homo = []
    l_hetero = []
    l_mean = []
    length = 40
    for i in range(length):
        feat = torch.randn(data.x.shape)
    
        pos_vectors, neg_vectors, auroc_homo, auroc_hetero, a, b, c = get_edge_vectors(features = feat, labels = labels, mp_index = edge_index, pos_edges = pos_labels, neg_edges = neg_labels)


        l_pos.append(pos_vectors)
        l_neg.append(neg_vectors)
        l_homo.append(auroc_homo)
        l_hetero.append(auroc_hetero)
    
    animate_trajectories(l_pos, l_neg, l_homo, l_hetero, 10, display = True)