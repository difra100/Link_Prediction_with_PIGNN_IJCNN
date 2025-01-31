import torch 

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.utils import is_undirected, to_undirected
from torch_sparse import SparseTensor
import time
from src.GRAFF import *
from src.animation_utils import *
from src.utils import *

def extract_probabilities(adj_matrix, coo_tensor):
    # Extract row and col indices from the COO tensor
    row_indices, col_indices = coo_tensor

    # Use the indices to extract the probabilities from the probability matrix
    probabilities = adj_matrix[row_indices, col_indices]

    return probabilities

def get_sparse_adjacency(adj_t, x):
    adj_t = SparseTensor.from_edge_index(adj_t,
                               sparse_sizes=(x.shape[0], x.shape[0])).to_device(
                                   x.device, non_blocking=True)
    adj_t = adj_t.to_symmetric()
    return adj_t

class Get_Metrics(Callback):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        # Compute the metrics
        train_loss = sum(
            pl_module.train_prop['loss']) / len(pl_module.train_prop['loss'])
        train_auroc = sum(
            pl_module.train_prop['AUROC'])/len(pl_module.train_prop['AUROC'])
        # train_acc100 = sum(
        #     pl_module.train_prop['HR@100']) / len(pl_module.train_prop['HR@100'])
        # train_acc20 = sum(
        #     pl_module.train_prop['HR@20']) / len(pl_module.train_prop['HR@20'])
        # train_acc1 = sum(
        #     pl_module.train_prop['HR@1']) / len(pl_module.train_prop['HR@1'])
        test_loss = sum(
            pl_module.test_prop['loss']) / len(pl_module.test_prop['loss'])
        test_auroc = sum(
            pl_module.test_prop['AUROC'])/len(pl_module.test_prop['AUROC'])
        # test_acc100 = sum(pl_module.test_prop['HR@100']) / \
        #     len(pl_module.test_prop['HR@100'])
        # test_acc20 = sum(pl_module.test_prop['HR@20']) / \
        #     len(pl_module.test_prop['HR@20'])
        # test_acc1 = sum(pl_module.test_prop['HR@1']) / \
        #     len(pl_module.test_prop['HR@1'])

        # Log the metrics
        pl_module.log(name='Loss on train', value=train_loss,
                      on_epoch=True, prog_bar=True, logger=True)
        pl_module.log(name='Loss on test', value=test_loss,
                      on_epoch=True, prog_bar=True, logger=True)
        pl_module.log(name='AUROC on train', value=train_auroc,
                      on_epoch=True, prog_bar=True, logger=True)
        pl_module.log(name='AUROC on test', value=test_auroc,
                      on_epoch=True, prog_bar=True, logger=True)
        
        # pl_module.log(name='HR@100 on train', value=train_acc100,
        #               on_epoch=True, prog_bar=True, logger=True)
        # pl_module.log(name='HR@100 on test', value=test_acc100,
        #               on_epoch=True, prog_bar=True, logger=True)

        # pl_module.log(name='HR@20 on train', value=train_acc20,
        #               on_epoch=True, prog_bar=True, logger=True)
        # pl_module.log(name='HR@20 on test', value=test_acc20,
        #               on_epoch=True, prog_bar=True, logger=True)

        # pl_module.log(name='HR@1 on train', value=train_acc1,
        #               on_epoch=True, prog_bar=True, logger=True)
        # pl_module.log(name='HR@1 on test', value=test_acc1,
        #               on_epoch=True, prog_bar=True, logger=True)
        pl_module.last_metric = test_auroc
    
        

        
        # Re-initialize the metrics
        pl_module.train_prop['loss'] = []
        pl_module.train_prop['HR@100'] = []
        pl_module.train_prop['HR@20'] = []
        pl_module.train_prop['HR@1'] = []
        pl_module.train_prop['AUROC'] = []


        pl_module.test_prop['loss'] = []
        pl_module.test_prop['HR@100'] = []
        pl_module.test_prop['HR@20'] = []
        pl_module.test_prop['HR@1'] = []
        pl_module.test_prop['AUROC'] = []

class TrainingModule(pl.LightningModule):

    def __init__(self, models, lr, wd, negatives, model_name = 'GRAFF', device = 'cpu', save_performance = False, readout_type = 'hadamard', display = False, lambda_ = 0, dataset_name = 'data'):
        super().__init__()
    
        self.model_name = model_name
        self.dataset_name = dataset_name
        if readout_type == 'gradient':
            self.dataset_name = f'{dataset_name}_grad'
        self.model_list = ['GRAFF', 'GAT', 'SAGE', 'GCN', 'mlp']
        self.dev = device
        self.save_performance = save_performance
        self.readout_type = readout_type

        if self.model_name in self.model_list or self.model_name == 'NCNC':
            self.model = models[0]
            self.predictor = models[1]
        
        
        

        elif self.model_name == 'disenlink':
            self.model = models

        elif self.model_name == 'ELPH':
            self.model = models

        
        self.display = display
        self.lambda_ = lambda_
        self.lr = lr
        self.wd = wd
        self.negatives = negatives

        self.best_metric = 0

        self.train_prop = {'loss': [], 'HR@100': [],
                           'HR@20': [], 'HR@1': [], 'AUROC': []}
        self.test_prop = {'loss': [], 'HR@100': [],
                          'HR@20': [], 'HR@1': [], 'AUROC': []}

    def training_step(self, batch, batch_idx):

        out = self.model(batch)

        pos_edge = batch.pos_edge_label_index

        if self.negatives > 1:
            neg_edge = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1]*self.negatives)]
        else:
            neg_edge = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])]

       

        if self.model_name in self.model_list:

            pos_out = get_readout_input(out, pos_edge, mp_edges = batch.edge_index, readout_type = self.readout_type, grad = True)

            

            neg_out = get_readout_input(out, neg_edge, mp_edges = batch.edge_index, readout_type = self.readout_type, grad = True)

            if self.readout_type == 'gradient':
                pos_out, pos_gradient = pos_out
                neg_out, neg_gradient = neg_out
            
            # print(d.shape)
            pos_pred = self.predictor(
                pos_out, training=False)
            
            neg_pred = self.predictor(
                neg_out, training=False)
        
        elif self.model_name == 'disenlink':
            
            _, adj_pred = out
            
            pos_pred = extract_probabilities(adj_pred, pos_edge)
            neg_pred = extract_probabilities(adj_pred, neg_edge)

        elif self.model_name == 'ELPH':

            out, hashes, cards = out



            pos_subgraph_features = self.model.elph_hashes.get_subgraph_features(pos_edge.T, hashes, cards)
            neg_subgraph_features = self.model.elph_hashes.get_subgraph_features(neg_edge.T, hashes, cards)
            pos_batch_node = out[pos_edge.T]
            neg_batch_node = out[neg_edge.T]

            pos_pred = self.model.predictor(pos_subgraph_features, pos_batch_node, None, training_mode = True).squeeze()
            neg_pred = self.model.predictor(neg_subgraph_features, neg_batch_node, None, training_mode = True).squeeze()

        elif self.model_name == 'NCNC':
            
            adj_t = get_sparse_adjacency(batch.edge_index, batch.x)

            pos_pred = self.predictor.multidomainforward(out, adj_t, pos_edge, [])
            neg_pred = self.predictor.multidomainforward(out, adj_t, neg_edge, [])




        if self.negatives != 0:

            loss = -torch.log(pos_pred + 1e-15).mean() - \
                torch.log(1 - neg_pred[:int(pos_pred.shape[0]*self.negatives)] + 1e-15).mean()
            
        else:

            loss = -torch.log(pos_pred + 1e-15).mean()

        auroc = compute_auroc(pos_pred, neg_pred[:pos_pred.shape[0]])

        if self.readout_type == 'gradient' and self.model_name in self.model_list:
            loss = loss - self.lambda_*neg_gradient


        self.train_prop['loss'].append(loss)
        self.train_prop['AUROC'].append(auroc)

        return loss

    def validation_step(self, batch, batch_idx):

        if len(self.train_prop['AUROC']) == 0 and self.model_name != 'GRAFF':
            print("Skip validation check....")
            return
        
        out = self.model(batch)

        pos_edge = batch.pos_edge_label_index

        if self.negatives > 1:
            neg_edge = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1]*self.negatives)]
        else:
            neg_edge = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])]

        if self.model_name in self.model_list:

            pos_out = get_readout_input(out, pos_edge, mp_edges = batch.edge_index, readout_type = self.readout_type)

            pos_pred = self.predictor(
                pos_out, training=False)

            neg_out = get_readout_input(out, neg_edge, mp_edges = batch.edge_index, readout_type = self.readout_type)

            neg_pred = self.predictor(
                neg_out, training=False)
        
        elif self.model_name == 'disenlink':
            
            _, adj_pred = out
            
            pos_pred = extract_probabilities(adj_pred, pos_edge)
            neg_pred = extract_probabilities(adj_pred, neg_edge)

        elif self.model_name == 'ELPH':

            out, hashes, cards = out

            pos_subgraph_features = self.model.elph_hashes.get_subgraph_features(pos_edge.T, hashes, cards)
            neg_subgraph_features = self.model.elph_hashes.get_subgraph_features(neg_edge.T, hashes, cards)
            pos_batch_node = out[pos_edge.T]
            neg_batch_node = out[neg_edge.T]

            pos_pred = self.model.predictor(pos_subgraph_features, pos_batch_node, None, training_mode = False).squeeze()
            neg_pred = self.model.predictor(neg_subgraph_features, neg_batch_node, None, training_mode = False).squeeze()
        
        elif self.model_name == 'NCNC':
            
            adj_t = get_sparse_adjacency(batch.edge_index, batch.x)

            pos_pred = self.predictor.multidomainforward(out, adj_t, pos_edge, [])
            neg_pred = self.predictor.multidomainforward(out, adj_t, neg_edge, [])


        
        if self.negatives != 0:
   
            loss = -torch.log(pos_pred + 1e-15).mean() - \
                torch.log(1 - neg_pred[:int(pos_pred.shape[0]*self.negatives)] + 1e-15).mean()
            
        else:

            loss = -torch.log(pos_pred + 1e-15).mean()

        auroc = compute_auroc(pos_pred, neg_pred[:pos_pred.shape[0]])

        self.test_prop['loss'].append(loss)
        self.test_prop['AUROC'].append(auroc)

        return loss
    
    def test_step(self, batch, batch_idx):
       
        start = time.time()

        if self.model_name in self.model_list or self.model_name == 'NCNC':
            out, list_embs = self.model(batch, save_embs = True)
        elif self.model_name == 'ELPH':
            out,  hashes, cards, list_embs = self.model(batch, save_embs = True)
        else:
            out = self.model(batch)
        


        pos_edge = batch.pos_edge_label_index

        
        if self.negatives > 1:
            neg_edge = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1]*self.negatives)]
        else:
            neg_edge = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])]

        if self.model_name in self.model_list:
            

            pos_out = get_readout_input(out, pos_edge, mp_edges = batch.edge_index, readout_type = self.readout_type, test = True, grad = True)
            

            neg_out = get_readout_input(out, neg_edge, mp_edges = batch.edge_index, readout_type = self.readout_type, test = True, grad = True)

            
            if self.readout_type == 'gradient':
                pos_out, pos_gradient = pos_out
                neg_out, neg_gradient = neg_out

            
            pos_pred = self.predictor(
                pos_out, training=False)
            
            neg_pred = self.predictor(
                neg_out, training=False)
            
            l_pos = []
            l_neg = []
            l_homo = []
            l_hetero = []
            l_mix_hard = []
            l_mix_easy = []
            l_exp_tot = []
            neg_median_gradients = []
            pos_median_gradients = []
            # Neg and Pos are the same length
            for step in range(len(list_embs)):
                pos_vectors, neg_vectors, exp_homo, exp_hetero, exp_mix_hard, exp_mix_easy, exp_tot, gradients = get_edge_vectors(features = list_embs[step].detach().cpu(), labels = batch.y.cpu(), mp_index = batch.edge_index.cpu(), pos_edges = pos_edge.cpu(), neg_edges = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])].cpu())
                l_pos.append(pos_vectors)
                l_neg.append(neg_vectors)
                l_homo.append(exp_homo)
                l_hetero.append(exp_hetero)
                l_mix_hard.append(exp_mix_hard)
                l_mix_easy.append(exp_mix_easy)
                l_exp_tot.append(exp_tot)
                pos_gradient, neg_gradient = gradients

                pos_median_gradients.append(pos_gradient.squeeze().cpu().numpy())
                neg_median_gradients.append(neg_gradient.squeeze().cpu().numpy())

      
            

            
            self.animation = animate_trajectories(l_pos, l_neg, l_homo, l_hetero, 200, display = self.display, model_name = self.model_name, dataset_name = self.dataset_name)

           
        
        elif self.model_name == 'disenlink':
            
            _, adj_pred = out
            
            pos_pred = extract_probabilities(adj_pred, pos_edge)
            neg_pred = extract_probabilities(adj_pred, neg_edge)
        
        elif self.model_name == 'ELPH':   

            pos_subgraph_features = self.model.elph_hashes.get_subgraph_features(pos_edge.T, hashes, cards)
            neg_subgraph_features = self.model.elph_hashes.get_subgraph_features(neg_edge.T, hashes, cards)
            pos_batch_node = out[pos_edge.T]
            neg_batch_node = out[neg_edge.T]

            pos_pred = self.model.predictor(pos_subgraph_features, pos_batch_node, None, training_mode = False).squeeze()
            neg_pred = self.model.predictor(neg_subgraph_features, neg_batch_node, None, training_mode = False).squeeze()

            l_pos = []
            l_neg = []
            l_homo = []
            l_hetero = []
            l_mix_hard = []
            l_mix_easy = []
            l_exp_tot = []
            neg_median_gradients = []
            pos_median_gradients = []
            # Neg and Pos are the same length

            for step in range(len(list_embs)):
                pos_vectors, neg_vectors, exp_homo, exp_hetero, exp_mix_hard, exp_mix_easy, exp_tot, gradients = get_edge_vectors(features = list_embs[step].detach().cpu(), labels = batch.y.cpu(), mp_index = batch.edge_index.cpu(), pos_edges = pos_edge.cpu(), neg_edges = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])].cpu())
                l_pos.append(pos_vectors)
                l_neg.append(neg_vectors)
                l_homo.append(exp_homo)
                l_hetero.append(exp_hetero)
                l_mix_hard.append(exp_mix_hard)
                l_mix_easy.append(exp_mix_easy)
                l_exp_tot.append(exp_tot)
                pos_gradient, neg_gradient = gradients

                pos_median_gradients.append(pos_gradient.squeeze().cpu().numpy())
                neg_median_gradients.append(neg_gradient.squeeze().cpu().numpy())

      
            
            self.animation = animate_trajectories(l_pos, l_neg, l_homo, l_hetero, 200, display = self.display)

        elif self.model_name == 'NCNC':
            
            adj_t = get_sparse_adjacency(batch.edge_index, batch.x)

            pos_pred = self.predictor.multidomainforward(out, adj_t, pos_edge, [])
            neg_pred = self.predictor.multidomainforward(out, adj_t, neg_edge, [])

            l_pos = []
            l_neg = []
            l_homo = []
            l_hetero = []
            l_mix_hard = []
            l_mix_easy = []
            l_exp_tot = []
            neg_median_gradients = []
            pos_median_gradients = []
            # Neg and Pos are the same length

            for step in range(len(list_embs)):
                pos_vectors, neg_vectors, exp_homo, exp_hetero, exp_mix_hard, exp_mix_easy, exp_tot, gradients = get_edge_vectors(features = list_embs[step].detach().cpu(), labels = batch.y.cpu(), mp_index = batch.edge_index.cpu(), pos_edges = pos_edge.cpu(), neg_edges = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])].cpu())
                l_pos.append(pos_vectors)
                l_neg.append(neg_vectors)
                l_homo.append(exp_homo)
                l_hetero.append(exp_hetero)
                l_mix_hard.append(exp_mix_hard)
                l_mix_easy.append(exp_mix_easy)
                l_exp_tot.append(exp_tot)
                pos_gradient, neg_gradient = gradients

                pos_median_gradients.append(pos_gradient.squeeze().cpu().numpy())
                neg_median_gradients.append(neg_gradient.squeeze().cpu().numpy())

      
            
            self.animation = animate_trajectories(l_pos, l_neg, l_homo, l_hetero, 200, display = self.display)


        if self.negatives != 0:

            loss = -torch.log(pos_pred + 1e-15).mean() - \
                torch.log(1 - neg_pred[:int(pos_pred.shape[0]*self.negatives)] + 1e-15).mean()
            
        else:
            
            loss = -torch.log(pos_pred + 1e-15).mean()

        auroc_homo, auroc_hetero, auroc_mix_hard, auroc_mix_easy = get_sub_aurocs(labels = batch.y.cpu(), pos_edges_vec = pos_pred.detach().cpu(), neg_edges_vec = neg_pred[:pos_pred.shape[0]].detach().cpu(), pos_edges = pos_edge.cpu(), neg_edges = batch.neg_edge_label_index[:, :int(pos_edge.shape[-1])].cpu())
        auroc = compute_auroc(pos_pred, neg_pred[:pos_pred.shape[0]])

        if self.display:
            dirichlet_list = []
            for step in range(len((list_embs))):
                dirichlet_energy_norm = get_dirichlet(list_embs[step], edges = batch.edge_index)
                dirichlet_list.append(dirichlet_energy_norm.item())
            
            plt.plot(dirichlet_list)
            #plt.show()

            


        if self.save_performance:
            self.pos = pos_pred.squeeze().detach().cpu().numpy()
            self.neg = neg_pred[:pos_pred.shape[0]].squeeze().detach().cpu().numpy()

        # Aurocs computations
        self.final_auroc = auroc
        self.auroc_homo = auroc_homo
        self.auroc_hetero = auroc_hetero 
        self.auroc_mix_hard = auroc_mix_hard
        self.auroc_mix_easy = auroc_mix_easy
        self.pos_gradients_distribution = np.array(pos_median_gradients)
        self.neg_gradients_distribution = np.array(neg_median_gradients)



        if self.model_name in self.model_list or self.model_name == 'ELPH' or self.model_name == 'NCNC':
            self.exp_homo = l_homo[-1]
            self.exp_hetero = l_hetero[-1]
            self.exp_mean = (self.exp_homo + self.exp_hetero) / 2
            self.exp_mix_hard = l_mix_hard[-1]
            self.exp_mix_easy = l_mix_easy[-1]
            self.exp_tot = l_exp_tot[-1]

        if self.model_name == 'GRAFF':
            self.max_depth = self.model.max_depth

        end = time.time()

        self.inference_time = end - start

        return loss

    def configure_optimizers(self):
        if self.model_name in self.model_list or self.model_name == 'NCNC':
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.predictor.parameters()), lr=self.lr, weight_decay=self.wd)
        elif self.model_name == 'disenlink':
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()), lr=self.lr, weight_decay=self.wd)
        
        elif self.model_name == 'ELPH':
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()), lr=self.lr, weight_decay=self.wd)
        
        return self.optimizer
    




