import numpy as np
import random
import torch
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.utils import homophily, degree
import os
from sklearn import metrics
SEED_list = [42, 123, 987, 555, 789, 999, 21022, 8888, 7777, 6543]

def set_seed(seed_value):
    # Set seed for NumPy
    # np.random.seed(seed_value)

    # # Set seed for Python's random module
    # random.seed(seed_value)

    # # Set seed for PyTorch
    # torch.manual_seed(seed_value)

    # # Set seed for GPU (if available)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)

    #     # Set the deterministic behavior for cudNN
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # Set seed for PyTorch Geometric
    torch_geometric.seed_everything(seed_value)

    # Set seed for PyTorch Lightning
    pl.seed_everything(seed_value, workers=True)
    print(f"{seed_value} have been correctly set!")
    # if torch.cuda.is_available():
    # #     # torch.cuda.manual_seed(seed_value)
    # #     # torch.cuda.manual_seed_all(seed_value)

    # #     # Set the deterministic behavior for cudNN
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


def get_D_k(dataset, class_label):
    # graph must be undirected
    # indices of a given class
    
    class_k_indices = torch.nonzero(dataset.y == class_label, as_tuple=False).squeeze()
    try:
        
        degree_k = 2*degree(dataset.edge_index[1]) # undirected case


        D_k = torch.sum(degree_k[class_k_indices])

        return D_k

    except IndexError:
        print(degree_k.shape[0])

def get_p_k(dataset, class_label):
    D_k = get_D_k(dataset, class_label)
    n_edges = dataset.edge_index.shape[1]
    p_k = D_k/(2*n_edges)

    return p_k

def adj_homophily(dataset):

    n_classes = torch.unique(dataset.y).shape[0]
    
    h_edge = homophily(edge_index = dataset.edge_index, y = dataset.y, method = 'edge')
    
    sum_p_k = sum([(get_p_k(dataset, k))**2 for k in range(n_classes)])

    h_adj = (h_edge - sum_p_k)/(1 - sum_p_k)

    return h_adj



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

    

def compute_auroc(positives_tensor, negatives_tensor):
    # Combine the positive and negative tensors
    all_tensor = torch.cat([positives_tensor, negatives_tensor])

    # Create the corresponding labels (1 for positives, 0 for negatives)
    labels = torch.cat([torch.ones_like(positives_tensor), torch.zeros_like(negatives_tensor)])

    # Calculate the AUROC using sklearn's roc_auc_score function
    auroc = metrics.roc_auc_score(labels.detach().cpu().numpy(), all_tensor.detach().cpu().numpy())

    return auroc

def get_dirichlet(emb, edges):
    x_s = emb[edges[0]]
    x_t = emb[edges[1]] 

    degrees = degree(edges[0], num_nodes=emb.shape[0], dtype=torch.float)

    # degrees = torch.where(degrees == 0, torch.tensor(1), degrees)

    feature_source = x_s / (torch.sqrt(degrees[edges[0]].view(-1, 1) + 1))
    feature_target = x_t / (torch.sqrt(degrees[edges[1]].view(-1, 1) + 1))

    diff = feature_target - feature_source
    
    norm  = torch.sum(diff*diff, dim = -1) 
    dirich = 0.5*torch.sum(norm)
    
    den = torch.norm(emb, p = 'fro')**2 + 1e-15
    print(den)

    dirich /= den
    
    return dirich