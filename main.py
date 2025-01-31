######## IMPORT EXTERNAL FILES ###########
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import gc

import argparse
import numpy as np
import wandb
from functools import partial

from sklearn import metrics

import pprint
import shutil

from src.GRAFF import *
from src.DisenLink import *

from src.ELPH import *

from src.config import *
from src.utils import *
from src.datamodule import DataModuleLP
from src.lightningmodule import *
from src.NCN.model import *
from src.NCN.util import *


set_determinism_the_old_way(deterministic = True)




parser = argparse.ArgumentParser()

parser.add_argument("-dataset_name", type = str)
parser.add_argument("-model_name", type = str)
parser.add_argument("-mode", type = str)
parser.add_argument("-wb", type = bool)
parser.add_argument("-sweep", type = bool)
parser.add_argument("-resume_sweep", type = str, default = '')
parser.add_argument("-eval", type = bool, default = False)
parser.add_argument("-device", type = str, default = device)
parser.add_argument("-num_data_splits", type = int, default = 1)
parser.add_argument("-save_performance", type = bool, default = False)
parser.add_argument("-get_gradients_distr", type = bool, default = False)
parser.add_argument("-get_inference_time", type = bool, default = False)


parser.add_argument("-count", type = int, default = 2000)
parser.add_argument("-pruned", type = bool, default = False)
parser.add_argument("-sweep_method", type = str, default = 'random')
parser.add_argument("-patience", type = int, default = 200)
parser.add_argument("-epochs", type = int, default = 5000)
parser.add_argument("-num_seeds", type = int, default = len(SEED_list))






parser.add_argument("-lr", type = float, default = lr)
parser.add_argument("-wd", type = float, default = wd)
parser.add_argument("-num_layers", type = int, default = 0)
parser.add_argument("-hidden_dim", type = int, default = 0)
parser.add_argument("-step", type = float, default = 1.)
parser.add_argument("-output_dim", type = int, default = 0)
parser.add_argument("-mlp_layer", type = int, default = 0)
parser.add_argument("-link_bias", type = bool, default = link_bias)
parser.add_argument("-dropout", type = float, default = 0.)
parser.add_argument("-input_dropout", type = float, default = 0)
parser.add_argument("-negatives", type = float, default = negatives)
parser.add_argument("-normalize", type = bool, default = False)
parser.add_argument("-aggregation", type = str, default = 'mean')
parser.add_argument("-heads", type = int, default = heads)
parser.add_argument("-pooling", type = str, default = pooling)
parser.add_argument("-gnn", type = bool, default = gnn)
parser.add_argument("-delta", type = bool, default = delta)
parser.add_argument("-iteration", type = bool, default = iteration)
parser.add_argument("-prev_step", type = bool, default = prev_step)
parser.add_argument("-threshold", type = float, default = threshold)
parser.add_argument("-readout_type", type = str, default = '')
parser.add_argument("-display", type = bool, default = False)
parser.add_argument("-lambda_", type = float, default = 0.)
parser.add_argument("-res", type = str, default = 'si')


# Disenlink parameters
parser.add_argument("-nfactor", type = int, default = nfactor)
parser.add_argument("-beta", type = float, default = beta)
parser.add_argument("-t", type = float, default = t)


# ELPH parameters
parser.add_argument("-max_hash_hops", type = int, default = 3)
parser.add_argument("-floor_sf", type = int, default = 0)
parser.add_argument("-minhash_num_perm", type = int, default = 128)
parser.add_argument("-use_zero_one", type = int, default = 0)
parser.add_argument("-use_feature", type = bool, default = True)
parser.add_argument("-feature_prop", type = str, default = 'residual')
parser.add_argument("-propagate_embeddings", type = bool, default = False)
parser.add_argument("-sign_k", type = int, default = 0)
parser.add_argument("-label_dropout", type = float, default = 0)
parser.add_argument("-hll_p", type = int, default = 8)


# NCNC parameters
# NCNC parameters
parser.add_argument('-depth', type=int, default=1)
parser.add_argument('-splitsize', type=int, default=-1)
parser.add_argument('-probscale', type=float, default=5.)
parser.add_argument('-proboffset', type=float, default=3.)
parser.add_argument('-trndeg', type=int, default=-1)
parser.add_argument('-tstdeg', type=int, default=-1)
parser.add_argument('-pt', type=float, default=0.5)
parser.add_argument('-learnpt', action="store_true")
parser.add_argument('-alpha', type=float, default=1.)
parser.add_argument('-hiddim', type=int, default=32)
parser.add_argument('-mplayers', type=int, default=1)
parser.add_argument('-gnndp', type=float, default=0.3)
parser.add_argument('-ln', action="store_true")
parser.add_argument('-model', choices=convdict.keys())
parser.add_argument('-jk', action="store_true")
parser.add_argument('-gnnedp', type=float, default=0.3)
parser.add_argument('-xdp', type=float, default=0.3)
parser.add_argument('-tdp', type=float, default=0.3)
parser.add_argument('-loadx', action="store_true")
parser.add_argument('-nnlayers', type=int, default=3)
parser.add_argument('-predp', type=float, default=0.3)
parser.add_argument('-preedp', type=float, default=0.3)
parser.add_argument('-lnnn', action="store_true")
parser.add_argument('-predictor', default="incn1cn1", choices=predictor_dict.keys())
parser.add_argument('-gnnlr', type=float, default=0.001)
parser.add_argument('-prelr', type=float, default=0.001)
parser.add_argument('-l2', type=float, default=1e-7)
parser.add_argument('-batch_size', type=int, default=1024)
parser.add_argument('-maskinput', action="store_true")
parser.add_argument('-use_xlin', action="store_true")
parser.add_argument('-tailact', action="store_true")


args = parser.parse_args()

lr = args.lr
wd = args.wd
num_layers = args.num_layers
hidden_dim = args.hidden_dim
step = args.step
output_dim = args.output_dim
mlp_layer = args.mlp_layer
link_bias = args.link_bias
dropout = args.dropout
input_dropout = args.input_dropout
negatives = args.negatives
resume_sweep = args.resume_sweep
eval = args.eval
device = args.device
sweep_method = args.sweep_method
normalize = args.normalize
save_performance = args.save_performance
heads = args.heads
aggregation = args.aggregation
num_data_splits = args.num_data_splits
pooling = args.pooling
gnn = args.gnn
delta = args.delta
iteration = args.iteration
prev_step = args.prev_step
threshold = args.threshold
readout_type = args.readout_type
display = args.display
patience = args.patience
num_seeds = args.num_seeds
get_gradients_distr = args.get_gradients_distr
get_inference_time = args.get_inference_time

if args.res == 'si':
    res = True
else:
    res = False

# Disenlink confs
nfactor = args.nfactor
beta = args.beta
t = args.t

# ELPH confs
max_hash_hops = args.max_hash_hops
floor_sf = args.floor_sf
minhash_num_perm = args.minhash_num_perm
use_zero_one = args.use_zero_one
use_feature = args.use_feature
feature_prop = args.feature_prop
propagate_embeddings = args.propagate_embeddings
sign_k = args.sign_k
label_dropout = args.label_dropout
args.feature_dropout = args.input_dropout
feature_dropout = args.feature_dropout
args.hidden_channels = args.hidden_dim
hidden_channels = args.hidden_channels
hll_p = args.hll_p
args.num_negs = args.negatives
num_negs = args.num_negs





dataset_name = args.dataset_name
model_name = args.model_name

wb = args.wb
sweep = args.sweep
mode = args.mode

count = args.count
pruned = args.pruned

model_list = ['GRAFF', 'SAGE', 'GCN', 'GAT', 'mlp']

SEED = 21022


epochs = args.epochs

hyperparameters['Dataset'] = dataset_name
hyperparameters['Model'] = model_name

sweep_config['method'] = sweep_method

if sweep:
    if model_name == '':
        sweep_config['parameters'] = parameters_dict['all_data']

    elif model_name == 'GRAFF':
        if not pruned:
            sweep_config['parameters'] = parameters_dict_GRAFF['all_data']
        else:
            sweep_config['parameters'] = parameters_dict_GRAFF[dataset_name]

    elif model_name == 'mlp':
        sweep_config['parameters'] = parameters_dict_mlp['all_data']

    elif model_name == 'SAGE':
        if not pruned:
            sweep_config['parameters'] = parameters_dict_SAGE['all_data']
        else:
            sweep_config['parameters'] = parameters_dict_SAGE[dataset_name]

    elif model_name == 'GAT':
        if not pruned:
            sweep_config['parameters'] = parameters_dict_GAT['all_data']
        else:
            sweep_config['parameters'] = parameters_dict_GAT[dataset_name]

    elif model_name == 'GCN':
        if not pruned:
            sweep_config['parameters'] = parameters_dict_GCN['all_data']
        else:
            sweep_config['parameters'] = parameters_dict_GCN[dataset_name]

    
    elif model_name == 'ELPH':
        sweep_config['parameters'] = parameters_dict_elph[dataset_name]

    elif model_name == 'NCNC':
        sweep_config['parameters'] = parameters_dict_NCNC['all_data']

    sweep_config['parameters']['dataset'] = {'values': [dataset_name] }
    
    if model_name != '':
        sweep_config['parameters']['model_name'] = {'values': [model_name] }

print("RES is: ", res)


def compute_runs_test():
    # Compute 10 runs of experiments not in a wandb sweep environment.
    metrics = []
    auroc_homo = []
    auroc_hetero = []
    auroc_mix_hard = []
    auroc_mix_easy = []


    if model_name in model_list or model_name == 'ELPH':
        exp_homo = []
        exp_hetero = []
        exp_mix_hard = []
        exp_mix_easy = []
        exp_mean = []
        exp_tot = []



    if save_performance:
        pos_array = np.array([])
        neg_array = np.array([])
 
    
    for n in range(num_data_splits):
        if num_data_splits > 1:
            train_data = torch.load("data/" + dataset_name + f"/train_data_{n+1}.pt")
            val_data = torch.load("data/" + dataset_name + f"/val_data_{n+1}.pt")
            test_data = torch.load("data/" + dataset_name + f"/test_data_{5}.pt")
            dataM = DataModuleLP(train_data.clone(), val_data.clone(), test_data.clone(), mode = mode)
            dataM.prepare_data()
            dataM.setup()
            print("New Data Split!")
        else:
            train_data = torch.load("data/" + dataset_name + "/train_data_0.pt")
            val_data = torch.load("data/" + dataset_name + "/val_data_0.pt")
            test_data = torch.load("data/" + dataset_name + "/test_data_0.pt")

            dataM = DataModuleLP(train_data.clone(), val_data.clone(), test_data.clone(), mode = mode)
            dataM.prepare_data()
            dataM.setup()

        input_features = train_data.x.shape[1]
        num_gpus = 1 if device != 'cpu' else 0

        if get_inference_time:
            times = []

        for run in range(num_seeds):
            print("SPLIT n° {}".format(n+1))

            print("RUN n° {}".format(run+1))
            set_seed(SEED_list[run])

            if model_name == 'GRAFF':

                model = PhysicsGNN_LP(input_features, hidden_dim,
                                        num_layers, input_dropout, normalize = normalize, step= step)
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], lr, wd, negatives, model_name, device, readout_type=readout_type, display = display, lambda_ = lambda_, dataset_name = dataset_name)
                
                exp_name = f"{SEED_list[run]}_{num_layers}_{input_dropout}_{hidden_dim}_{output_dim}_{mlp_layer}_{dropout}_{lr}_{wd}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"
        
            elif model_name == 'mlp':
     
                model = MLP(input_features, hidden_dim, n_layers = num_layers, device=device, dropout_prob= dropout)

                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], lr, wd, negatives, model_name, device = device, readout_type=readout_type, display = display, lambda_ = lambda_, dataset_name = dataset_name)
                
                exp_name = f"{SEED_list[run]}_{num_layers}_{input_dropout}_{hidden_dim}_{output_dim}_{mlp_layer}_{dropout}_{lr}_{wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{model_name}/"
            
            elif model_name == 'SAGE':
                model = GNN_LP(input_features, hidden_dim, num_layers, GNN_args=[SAGEConv, 'SAGE', aggregation, res])
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], lr, wd, negatives, model_name, device, readout_type=readout_type, display = display, lambda_ = lambda_, dataset_name = dataset_name)
                
                exp_name = f"{SEED_list[run]}_{num_layers}_{input_dropout}_{aggregation}_{hidden_dim}_{output_dim}_{mlp_layer}_{dropout}_{lr}_{wd}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"
            
            elif model_name == 'GAT':
                model = GNN_LP(input_features, hidden_dim, num_layers, GNN_args=[GATConv, 'GAT', heads, res])
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], lr, wd, negatives, model_name, device, readout_type=readout_type, display = display, lambda_ = lambda_, dataset_name = dataset_name)
                
                exp_name = f"{SEED_list[run]}_{num_layers}_{input_dropout}_{heads}_{hidden_dim}_{output_dim}_{mlp_layer}_{dropout}_{lr}_{wd}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"
            
            elif model_name == 'GCN':
                model = GNN_LP(input_features, hidden_dim, num_layers, GNN_args=[GCNConv, 'GCN', res])

                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], lr, wd, negatives, model_name, device, readout_type=readout_type, display = display, lambda_ = lambda_, dataset_name = dataset_name)
                
                exp_name = f"{SEED_list[run]}_{num_layers}_{input_dropout}_{hidden_dim}_{output_dim}_{mlp_layer}_{dropout}_{lr}_{wd}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"
            
            elif model_name == 'disenlink':
                model = Disentangle(input_features, hidden_dim, output_dim,nfactor,beta,t)
                pl_training_module = TrainingModule(model, lr, wd, negatives, model_name=model_name, device = device, display = display, lambda_ = lambda_, dataset_name = dataset_name) 
                
                exp_name = f"{SEED_list[run]}_{num_layers}_{input_dropout}_{hidden_dim}_{output_dim}_{nfactor}_{mlp_layer}_{dropout}_{lr}_{wd}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"
            
            elif model_name == 'ELPH':
        
                model = ELPH(args, input_features, device=device)
                pl_training_module = TrainingModule(model, lr, wd, num_negs, model_name=model_name, device = device, display = display, dataset_name = dataset_name)    
                
                exp_name = f"{SEED_list[run]}_{max_hash_hops}_{feature_dropout}_{hidden_channels}_{num_negs}_{label_dropout}_{lr}_{wd}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"

            elif model_name == 'NCNC':

                predfn = predictor_dict[args.predictor]
                if args.predictor == "incn1cn1":
                    predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
                
                model = GCN(input_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, -1,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
                
                predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                                args.predp, args.preedp, args.lnnn).to(device)
                
                pl_training_module = TrainingModule([model, predictor], lr, wd, num_negs, model_name=model_name, device = device, display = display, dataset_name = dataset_name)    
                
                exp_name = f"{SEED_list[run]}_{args.hiddim}_{args.mplayers}_{args.gnndp}_{args.ln}_{args.res}_{lr}_{wd}_{args.xdp}_{args.tdp}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"

            model_params = get_n_params(model)
            print("MODEL number of parameters: ", model_params)

            tot_dir = prefix + exp_name + '/'
            shutil.rmtree(prefix, ignore_errors=True)
            os.makedirs(prefix, exist_ok=True)
        
            checkpoint_callback = ModelCheckpoint(dirpath = tot_dir,
                save_top_k=1, monitor="AUROC on test", mode="max")
            early_stop = EarlyStopping(monitor='AUROC on test', patience=patience, mode="max")
            compute_metrics = Get_Metrics()

            if wb:
                wandb_logger = WandbLogger(
                    project=project_name, name=exp_name, config=hyperparameters)
                trainer = pl.Trainer(
                    max_epochs=epochs,  # maximum number of epochs.
                    devices = 1, accelerator = 'gpu',  # the number of gpus we have at our disposal.
                    default_root_dir=prefix, callbacks=[compute_metrics, early_stop, checkpoint_callback],
                    logger=wandb_logger,
                    enable_checkpointing=True, deterministic = True if device == 'cpu' else False
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=epochs,  # maximum number of epochs.
                    devices = 1, accelerator = 'gpu',  # the number of gpus we have at our disposal.
                    default_root_dir=prefix, callbacks=[compute_metrics, early_stop, checkpoint_callback],
                    enable_checkpointing=True, deterministic = True if device == 'cpu' else False
                )

            trainer.fit(model=pl_training_module, datamodule=dataM)

            # prints path to the best model's checkpoint
            # print("Best model path is:", checkpoint_callback.best_model_path)
            # # and prints it score
            # print("Best model score is:\n", checkpoint_callback.best_model_score)

            set_seed(SEED_list[run])

            if model_name == 'GRAFF':

                model = PhysicsGNN_LP(input_features, hidden_dim,
                                        num_layers, input_dropout, normalize = normalize, step=step)
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
            
            elif model_name == 'mlp':
     
                model = MLP(input_features, hidden_dim, n_layers = num_layers, device=device, dropout_prob= dropout)

                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
             
            elif model_name == 'SAGE':
                model = GNN_LP(input_features, hidden_dim, num_layers, GNN_args=[SAGEConv, 'SAGE', aggregation, res])
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
            
            elif model_name == 'GAT':
                model = GNN_LP(input_features, hidden_dim, num_layers, GNN_args=[GATConv, 'GAT', heads, res])
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
            
            elif model_name == 'GCN':
                model = GNN_LP(input_features, hidden_dim, num_layers, GNN_args=[GCNConv, 'GCN', res])
                predictor = LinkPredictor_(
                hidden_dim, output_dim, mlp_layer, link_bias, dropout, device=device)
            elif model_name == 'disenlink':
                model = Disentangle(input_features, hidden_dim, output_dim,nfactor,beta,t)
            
            elif model_name == 'ELPH':
                model = ELPH(args, input_features, device=device)

            elif model_name == 'NCNC':

                predfn = predictor_dict[args.predictor]
                if args.predictor == "incn1cn1":
                    predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
                
                model = GCN(input_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, -1,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
                
                predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                                args.predp, args.preedp, args.lnnn).to(device)
                
                
            if model_name in model_list or model_name == 'NCNC':
                test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            models=[model, predictor], lr=lr, wd=wd, negatives = negatives, model_name = model_name, device = device, save_performance = save_performance, readout_type = readout_type, display = display, lambda_ = lambda_, dataset_name = dataset_name)
            elif model_name == 'disenlink':
                test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            models=model, lr=lr, wd=wd, negatives = negatives, model_name = model_name, device = device, save_performance = save_performance, display = display, lambda_ = lambda_, dataset_name = dataset_name)
            
            elif model_name == 'ELPH':
                test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            models=model, lr=lr, wd=wd, negatives = num_negs, model_name = model_name, device = device, save_performance = save_performance, display = display, dataset_name = dataset_name)


            trainer.test(test_model, dataloaders=dataM.test_dataloader())

            metrics.append(round(test_model.final_auroc, 4))
            auroc_homo.append(round(test_model.auroc_homo, 4))
            auroc_hetero.append(round(test_model.auroc_hetero, 4))
            auroc_mix_hard.append(round(test_model.auroc_mix_hard, 4))
            auroc_mix_easy.append(round(test_model.auroc_mix_easy, 4))


            print("Test model score is:", test_model.final_auroc)
            print("Auroc Homo is:", test_model.auroc_homo)
            print("Auroc Hetero is:", test_model.auroc_hetero)
            print("Auroc Mix Hard is:", test_model.auroc_mix_hard)
            print("Auroc Mix Easy is:", test_model.auroc_mix_easy)
            
            if get_gradients_distr:
                directory_name = f'gradient_distributions/{model_name}/{dataset_name}'
                os.makedirs(directory_name, exist_ok=True)
                pos_path = directory_name + '/pos_distribution.npy'
                neg_path = directory_name + '/neg_distribution.npy'
                np.save(pos_path, test_model.pos_gradients_distribution)
                np.save(neg_path, test_model.neg_gradients_distribution)
            
            if get_inference_time:
                times.append(test_model.inference_time)
                
                

            if model_name in model_list or model_name == 'ELPH':
                print("Exp Homo is:", test_model.exp_homo)
                print("Exp Hetero is:", test_model.exp_hetero)
                print("Exp Mix Hard is:", test_model.exp_mix_hard)
                print("Exp Mix Easy is:", test_model.exp_mix_easy)
                print("Exp Mean is:", test_model.exp_mean)
                print("Exp Tot. is:", test_model.exp_tot)


            
                exp_homo.append(round(test_model.exp_homo, 4))
                exp_hetero.append(round(test_model.exp_hetero, 4))
                exp_mix_hard.append(round(test_model.exp_mix_hard, 4))
                exp_mix_easy.append(round(test_model.exp_mix_easy, 4))
                exp_mean.append(round(test_model.exp_mean, 4))
                exp_tot.append(round(test_model.exp_tot, 4))


            


            if save_performance:
                pos_array = np.concatenate((pos_array, test_model.pos))
                neg_array = np.concatenate((neg_array, test_model.neg))


            
            del model
            
            del pl_training_module
            del trainer
            del test_model
            del checkpoint_callback
            del early_stop
            del compute_metrics

            if model_name in model_list:
                del predictor

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if get_inference_time:
        directory_name = f'inference_time/{model_name}_{readout_type}/{dataset_name}'
        os.makedirs(directory_name, exist_ok=True)
        times_path = directory_name + '/times.npy'
        parameters = directory_name + '/num_parameters.npy'
        np.save(times_path, np.array(times))
        np.save(parameters, np.array([model_params]))

    if save_performance:
        
        if res:
            if readout_type == 'hadamard':
                path = './performance/' + dataset_name + '/' + model_name + '/'

            elif readout_type == 'gradient':
                path = './performance_gradient/' + dataset_name + '/' + model_name + '/'
        else:
            if readout_type == 'hadamard':
                path = './performance_nores/' + dataset_name + '/' + model_name + '/'

            elif readout_type == 'gradient':
                path = './performance_gradient_nores/' + dataset_name + '/' + model_name + '/'


        positive_path = path + 'positives.npy'
        negative_path = path + 'negatives.npy'
        aurocs_path = path + 'AUROC.npy'
        aurocs_homo_path = path + 'AUROC_homo.npy'
        aurocs_hetero_path = path + 'AUROC_hetero.npy'
        aurocs_mix_hard_path = path + 'AUROC_mix_hard_path.npy'
        aurocs_mix_easy_path = path + 'AUROC_mix_easy_path.npy'


        auroc_array = np.array(metrics)
        auroc_homo_array = np.array(auroc_homo)
        auroc_hetero_array = np.array(auroc_hetero)
        auroc_mix_hard_array = np.array(auroc_mix_hard)
        auroc_mix_easy_array = np.array(auroc_mix_easy)




        dir = os.listdir(path)
        np.save(positive_path, pos_array)
        np.save(negative_path, neg_array)
        np.save(aurocs_path, auroc_array)
        np.save(aurocs_homo_path, auroc_homo_array)
        np.save(aurocs_hetero_path, auroc_hetero_array)
        np.save(aurocs_mix_hard_path, auroc_mix_hard_array)
        np.save(aurocs_mix_easy_path, auroc_mix_easy_array)

        if model_name in model_list or model_name == 'ELPH':
            exp_homo_path = path + 'exp_homo.npy'
            exp_hetero_path = path + 'exp_hetero.npy'
            exp_mean_path = path + 'exp_mean.npy'
            exp_mix_hard_path = path + 'exp_mix_hard.npy'
            exp_mix_easy_path = path + 'exp_mix_easy.npy'
            exp_tot_path = path + 'exp_tot.npy'

            exp_homo_array = np.array(exp_homo)
            exp_hetero_array = np.array(exp_hetero)
            exp_mean_array = np.array(exp_mean)
            exp_mix_hard_array = np.array(exp_mix_hard)
            exp_mix_easy_array = np.array(exp_mix_easy)
            exp_tot_array = np.array(exp_tot)

            np.save(exp_homo_path, exp_homo_array)
            np.save(exp_hetero_path, exp_hetero_array)
            np.save(exp_mean_path, exp_mean_array)
            np.save(exp_mix_hard_path, exp_mix_hard_array)
            np.save(exp_mix_easy_path, exp_mix_easy_array)
            np.save(exp_tot_path, exp_tot_array)



    if model_name in model_list:

        return metrics, exp_homo, exp_hetero, exp_mean, exp_mix_hard, exp_mix_easy, exp_tot
    
    else:

        return metrics


def compute_runs(config):
    # compute 10 runs of experiments for each configuration in the sweep.
    metrics = []
    if config.model_name in model_list:
        exp_homo = []
        exp_hetero = []
        exp_mean = []
    SEED_list = [42, 123, 987, 555, 789] #, 999, 21022, 8888, 7777, 6543]
    for n in range(num_data_splits):

        if num_data_splits > 1:
            train_data = torch.load("data/" + dataset_name + f"/train_data_{n+1}.pt")
            val_data = torch.load("data/" + dataset_name + f"/val_data_{n+1}.pt")
            test_data = torch.load("data/" + dataset_name + f"/test_data_{5}.pt")
            dataM = DataModuleLP(train_data.clone(), val_data.clone(), test_data.clone(), mode = mode)
            dataM.prepare_data()
            dataM.setup()
            print("New Data Split!")
        else:
            train_data = torch.load("data/" + dataset_name + "/train_data_0.pt")
            val_data = torch.load("data/" + dataset_name + "/val_data_0.pt")
            test_data = torch.load("data/" + dataset_name + "/test_data_0.pt")



            dataM = DataModuleLP(train_data.clone(), val_data.clone(), test_data.clone(), mode = mode)
            dataM.prepare_data()
            dataM.setup()
        
        input_features = train_data.x.shape[1]
        num_gpus = 1 if device != 'cpu' else 0
        
        for run in range(len(SEED_list)):
            print(f"SPLIT n° {n+1}")
            print("RUN n° {}".format(run+1))
            set_seed(SEED_list[run])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            
            if config.model_name == 'GRAFF':
                model = PhysicsGNN_LP(input_features, config.hidden_dim,
                                        config.num_layers, config.input_dropout, normalize = config.normalize, step= config.step)
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], config.lr, config.wd, config.negatives, config.model_name, device = device, readout_type=config.readout_type, display = display, lambda_ = config.lambda_)
                
                exp_name = f"{SEED_list[run]}_{config.num_layers}_{config.input_dropout}_{config.hidden_dim}_{config.output_dim}_{config.mlp_layer}_{config.dropout}_{config.lr}_{config.wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{config.model_name}/"
            
            elif config.model_name == 'mlp':
     
                model = MLP(input_features, config.hidden_dim, n_layers=config.num_layers, device=device, dropout_prob=config.dropout)

                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], config.lr, config.wd, config.negatives, config.model_name, device = device, readout_type=config.readout_type, display = display, lambda_ = config.lambda_)
                
                exp_name = f"{SEED_list[run]}_{config.num_layers}_{config.input_dropout}_{config.hidden_dim}_{config.output_dim}_{config.mlp_layer}_{config.dropout}_{config.lr}_{config.wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{config.model_name}/"
    
            elif config.model_name == 'SAGE':
                model = GNN_LP(input_features, config.hidden_dim, config.num_layers, GNN_args=[SAGEConv, 'SAGE', config.aggregation, config.res])
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], config.lr, config.wd, config.negatives, config.model_name, device = device, readout_type=config.readout_type, display = display, lambda_ = config.lambda_)
                
                exp_name = f"{SEED_list[run]}_{config.num_layers}_{config.input_dropout}_{config.aggregation}_{config.hidden_dim}_{config.output_dim}_{config.mlp_layer}_{config.dropout}_{config.lr}_{config.wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{config.model_name}/"
            
            elif config.model_name == 'GAT':
                model = GNN_LP(input_features, config.hidden_dim, config.num_layers, GNN_args=[GATConv, 'GAT', config.heads, config.res])
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], config.lr, config.wd, config.negatives, config.model_name, device = device, readout_type=config.readout_type, display = display, lambda_ = config.lambda_)
                
                exp_name = f"{SEED_list[run]}_{config.num_layers}_{config.input_dropout}_{config.heads}_{config.hidden_dim}_{config.output_dim}_{config.mlp_layer}_{config.dropout}_{config.lr}_{config.wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{config.model_name}/"
            
            elif config.model_name == 'GCN':
                model = GNN_LP(input_features, config.hidden_dim, config.num_layers, GNN_args=[GCNConv, 'GCN', config.res])

                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
                pl_training_module = TrainingModule(
                    [model, predictor], config.lr, config.wd, config.negatives, config.model_name, device = device, readout_type=config.readout_type, display = display, lambda_ = config.lambda_)
                
                exp_name = f"{SEED_list[run]}_{config.num_layers}_{config.input_dropout}_{config.hidden_dim}_{config.output_dim}_{config.mlp_layer}_{config.dropout}_{config.lr}_{config.wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{config.model_name}/"
            
            elif config.model_name == 'disenlink':
                model = Disentangle(input_features, config.hidden_dim, config.output_dim, config.nfactor,beta,t)
                pl_training_module = TrainingModule(model, config.lr, config.wd, config.negatives, model_name=config.model_name, device = device, display = display, lambda_ = config.lambda_) 
                
                exp_name = f"{SEED_list[run]}_{config.hidden_dim}_{config.output_dim}_{config.nfactor}_{config.lr}_{config.wd}_{config.negatives}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{config.model_name}/"
            
            
            elif model_name == 'ELPH':
        
                model = ELPH(config, input_features, device=device)
                pl_training_module = TrainingModule(model, config.lr, config.wd, config.num_negs, model_name=model_name, device = device, display = display)    
                
                exp_name = f"{SEED_list[run]}_{config.max_hash_hops}_{config.feature_dropout}_{config.hidden_channels}_{config.num_negs}_{config.label_dropout}_{config.lr}_{config.wd}/"
                prefix = f"Sweeps/{dataset_name}_sweeps/{model_name}/"
            
            elif config.model_name == 'NCNC':

                predfn = predictor_dict[config.predictor]
                # if args.predictor == "incn1cn1":
                #     predfn = partial(predfn, depth=config.depth, splitsize=config.splitsize, scale=config.probscale, offset=config.proboffset, trainresdeg=config.trndeg, testresdeg=config.tstdeg, pt=config.pt, learnablept=config.learnpt, alpha=config.alpha)
                
                model = GCN(input_features, config.hiddim, config.hiddim, config.mplayers,
                    config.gnndp, config.ln, config.res, -1,
                    config.model, config.jk, config.gnnedp,  xdropout=config.xdp, taildropout=config.tdp, noinputlin=config.loadx).to(device)
                
                predictor = predfn(config.hiddim, config.hiddim, 1, config.nnlayers,
                                config.predp, config.preedp, config.lnnn).to(device)
                
                pl_training_module = TrainingModule([model, predictor], lr, wd, num_negs, model_name=model_name, device = device, display = display, dataset_name = dataset_name)    
                
                exp_name = f"{SEED_list[run]}_{config.hiddim}_{config.mplayers}_{config.gnndp}_{config.ln}_{config.res}_{lr}_{wd}_{config.xdp}_{config.tdp}/"
                prefix = f"Tests/{dataset_name}_Tests/{model_name}/"

   

            tot_dir = prefix + exp_name + '/'
            shutil.rmtree(prefix, ignore_errors=True)
            os.makedirs(prefix, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(dirpath = tot_dir,
                save_top_k=1, monitor="AUROC on test", mode="max")
            early_stop = EarlyStopping(monitor='AUROC on test', patience=patience, mode="max")
            compute_metrics = Get_Metrics()

            # wandb_logger = WandbLogger(project=project_name, name=exp_name, config=hyperparameters)
            print("NUM_GPUS: ", num_gpus)
            trainer = pl.Trainer(
                max_epochs=epochs,  # maximum number of epochs.
                devices = 1, accelerator = 'gpu',  # the number of gpus we have at our disposal.
                default_root_dir=prefix, callbacks=[compute_metrics, early_stop, checkpoint_callback],
            # logger=wandb_logger,
                enable_checkpointing=True
            )


            
            trainer.fit(model=pl_training_module, datamodule=dataM)

            
            set_seed(SEED_list[run])

            if config.model_name == 'GRAFF':

                model = PhysicsGNN_LP(input_features, config.hidden_dim,
                                        config.num_layers, config.input_dropout, normalize = config.normalize, step= config.step)
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
            elif config.model_name == 'mlp':
     
                model = MLP(input_features, config.hidden_dim, n_layers=config.num_layers, device=device, dropout_prob=config.dropout)

                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
        
                
            elif config.model_name == 'SAGE':
                model = GNN_LP(input_features, config.hidden_dim, config.num_layers, GNN_args=[SAGEConv, 'SAGE', config.aggregation, config.res])
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
            
            elif config.model_name == 'GAT':
                model = GNN_LP(input_features, config.hidden_dim, config.num_layers, GNN_args=[GATConv, 'GAT', config.heads, config.res])
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
            
            elif config.model_name == 'GCN':
                model = GNN_LP(input_features, config.hidden_dim, config.num_layers, GNN_args=[GCNConv, 'GCN', config.res])
                predictor = LinkPredictor_(
                config.hidden_dim, config.output_dim, config.mlp_layer, link_bias, config.dropout, device=device)
            elif config.model_name == 'disenlink':
                model = Disentangle(input_features, config.hidden_dim, config.output_dim, config.nfactor, beta, t)
            
            elif model_name == 'ELPH':
                model = ELPH(config, input_features, device=device)
            elif config.model_name == 'NCNC':

                predfn = predictor_dict[config.predictor]
                # if args.predictor == "incn1cn1":
                #     predfn = partial(predfn, depth=config.depth, splitsize=config.splitsize, scale=config.probscale, offset=config.proboffset, trainresdeg=config.trndeg, testresdeg=config.tstdeg, pt=config.pt, learnablept=config.learnpt, alpha=config.alpha)
                
                model = GCN(input_features, config.hiddim, config.hiddim, config.mplayers,
                    config.gnndp, config.ln, config.res, -1,
                    config.model, config.jk, config.gnnedp,  xdropout=config.xdp, taildropout=config.tdp, noinputlin=config.loadx).to(device)
                
                predictor = predfn(config.hiddim, config.hiddim, 1, config.nnlayers,
                                config.predp, config.preedp, config.lnnn).to(device)
                
                
            print("MODEL number of parameters: ", get_n_params(model))


            if config.model_name in model_list or config.model_name == 'NCNC':
                test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            models=[model, predictor], lr=config.lr, wd=config.wd, negatives = config.negatives, model_name = config.model_name, device = device, readout_type=config.readout_type, display = display, lambda_ = config.lambda_)
            elif config.model_name == 'disenlink':
                test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            models=model, lr=config.lr, wd=config.wd, negatives = config.negatives, model_name = config.model_name, device = device, display = display, lambda_ = config.lambda_)

            elif model_name == 'ELPH':
                test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            models=model, lr=config.lr, wd=config.wd, negatives = config.num_negs, model_name = model_name, device = device, display = display)


            trainer.test(test_model, dataloaders=dataM.val_dataloader())

            if config.model_name == 'GRAFF':
                wandb.log({"Max Depth": test_model.max_depth})

            
            metrics.append(round(test_model.final_auroc, 4))


            print("CHECKPOINT MODEL SCORE: "*10, checkpoint_callback.best_model_score)
            print("Test model score is:", test_model.final_auroc)
            


            if config.model_name in model_list:

                exp_homo.append(round(test_model.exp_homo, 4))
                exp_hetero.append(round(test_model.exp_hetero, 4))
                exp_mean.append(round(test_model.exp_mean, 4))

            del model
            del pl_training_module
            del trainer
            del test_model
            del checkpoint_callback
            del early_stop
            del compute_metrics

            if config.model_name in model_list:
                del predictor

            gc.collect()

    if config.model_name in model_list:
        return metrics, exp_homo, exp_hetero, exp_mean
    else:
        return metrics


def sweep_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, resume = True if resume_sweep != '' else False):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        if config.model_name in model_list:
            metrics, exp_homo, exp_hetero, exp_mean = compute_runs(config = config)
        else:
            metrics = compute_runs(config = config)

        metrics = np.array(metrics)
        mean, std = np.mean(metrics), np.std(metrics)
        print("AUROC, MEAN AND STANDARD DEVIATION: ", mean, std)

        wandb.log({"AUROC on test (Mean)": mean, "AUROC on test (Std.)": std})
        wandb.log({"AUROC scores": metrics})

        if config.model_name in model_list:
            
            exp_homo = np.array(exp_homo)
            mean, std = np.mean(exp_homo), np.std(exp_homo)
            print("Explainability Homophily, MEAN AND STANDARD DEVIATION: ", mean, std)

            wandb.log({"Explainability Homophily on test (mean)": mean, "Explainability Homophily on test (Std.)": std})

            exp_hetero = np.array(exp_hetero)
            mean, std = np.mean(exp_hetero), np.std(exp_hetero)
            print("Explainability Heterophily, MEAN AND STANDARD DEVIATION: ", mean, std)
            wandb.log({"Explainability Heterophily on test (mean)": mean, "Explainability Homophily on test (Std.)": std})


            exp_mean = np.array(exp_mean)
            mean, std = np.mean(exp_mean), np.std(exp_mean)
            print("Explainability Mean, MEAN AND STANDARD DEVIATION: ", mean, std)
            wandb.log({"Explainability Mean on test (mean)": mean, "Explainability Mean on test (Std.)": std})






if wb:
    wandb.login()


# Evaluation of the performance averaged on all the seed on test or val
if (not sweep or mode == 'test') and (eval):
    # Evaluate one configuration on test with 10 random seed
    
    if model_name in model_list:
        metrics, exp_homo, exp_hetero, exp_mean, exp_mix_hard, exp_mix_easy, exp_tot = compute_runs_test()
    else: 
        metrics = compute_runs_test()

    
    
    metrics = np.array(metrics)
    mean, std = np.mean(metrics), np.std(metrics)
    print("AUROC, MEAN AND STANDARD DEVIATION: ", mean, std)

    if model_name in model_list:
        exp_homo = np.array(exp_homo)
        mean, std = np.mean(exp_homo), np.std(exp_homo)
        print("Explainability Homophily, MEAN AND STANDARD DEVIATION: ", mean, std)

        exp_hetero = np.array(exp_hetero)
        mean, std = np.mean(exp_hetero), np.std(exp_hetero)
        print("Explainability Heterophily, MEAN AND STANDARD DEVIATION: ", mean, std)


        exp_mean = np.array(exp_mean)
        mean, std = np.mean(exp_mean), np.std(exp_mean)
        print("Explainability Mean, MEAN AND STANDARD DEVIATION: ", mean, std)

        exp_mix_hard = np.array(exp_mix_hard)
        mean, std = np.mean(exp_mix_hard), np.std(exp_mix_hard)
        print("Explainability Mix Hard, MEAN AND STANDARD DEVIATION: ", mean, std)

        exp_mix_easy = np.array(exp_mix_easy)
        mean, std = np.mean(exp_mix_easy), np.std(exp_mix_easy)
        print("Explainability Mix Easy, MEAN AND STANDARD DEVIATION: ", mean, std)

        exp_tot = np.array(exp_tot)
        mean, std = np.mean(exp_tot), np.std(exp_tot)
        print("Explainability Tot., MEAN AND STANDARD DEVIATION: ", mean, std)

elif mode == 'hp' and sweep:

    if resume_sweep != '':
        sweep_id = resume_sweep
        print("RESUMED PAST SWEEP....")
    else:

        sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity_name)
    pprint.pprint(sweep_config)

    
    wandb.agent(sweep_id, sweep_train, count = count, project = project_name, entity= entity_name)

    wandb.finish()





