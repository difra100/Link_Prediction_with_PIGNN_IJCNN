import torch

project_name = 'Link-Prediction-with-PBGNN-single'
entity_name = 'difra00'



lr = 0.01 #
wd = 0 #
num_layers = 50
step = 1
hidden_dim = 128 # 
output_dim = 64
mlp_layer = 1
link_bias = True
dropout = 0.5 #
input_dropout = 0.3
negatives = 8 #
aggregation = 'mean'
pooling = 'vpa'
delta = False
iteration = True
gnn = False
prev_step = True
threshold = 0.1
lambda_ = 0


heads = 1
normalize = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Disenlink Confs
nfactor = 8
beta = 0.9
t = 1





hyperparameters = {'learning rate': lr,
                   'weight decay': wd,
                   'output_dim': output_dim,
                   'mlp_layers': mlp_layer,
                   'link_bias': link_bias,
                   'decoder dropout': dropout,
                   'encoder dropout': input_dropout,
                   'N° Hidden layer': num_layers,
                   'Hidden dimension in GRAFF': hidden_dim,
                   'ODE step': step,
                   'Negative ratio': negatives}


# Set sweep = True in the notebook when is required to do the hyperparameter tuning with sweep #

sweep_config = {
    'method': 'random'
}

sweep_config['metric'] = {'name': 'AUROC on test (Mean)',
                          'goal': 'maximize'
                         }


parameters_dict = { # 559872 possible confs
    'all_data':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'model_name': {
        'values': ['GRAFF']
    },
    'wd': {
        'values': [0]
    },
    'lambda_': {
        'values': [0]
    },
    'step': {
        'values': [0.5]
    },
    'num_layers': {
        'values': [1, 3, 5, 7]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0, 1]
    },
    'dropout': {
        'values': [0]
    },
    'input_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'negatives': {
        'values': [1, 2, 4]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['gradient']
    }, 
    'heads': {
        'values': [1]
    }, 
    'aggregation': {
        'values': ['mean']
    }
    }
}


parameters_dict_mlp = { # 559872 possible confs
    'all_data':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-3]
    },
    'num_layers': {
        'values': [1, 3, 5, 7]
    },
    'output_dim': {
        'values': [64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0.3]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [0.25, 0, 4, 8]
    }, 
    'readout_type': {
        'values': ['hadamard']
    }, 
    'lambda_': {
        'values': [0]
    },
    'normalize': {
        'values': [True]
    },
    "res": {
        'values': [True]
    }
    }
    
}


parameters_dict_elph = { # 559872 possible confs
    
    'tolokers':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_channels': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-3]
    },
    'max_hash_hops': {
        'values': [3]
    },
    'output_dim': {
        'values': [64]
    },
    'label_dropout': {
        'values': [0.3]
    },
    'feature_dropout': {
        'values': [0.3]
    },
    'num_negs': {
        'values': [0.25, 0, 4, 8]
    }, 
    'readout_type': {
        'values': ['hadamard']
    },
    'lambda_': {
        'values': [0]
    }, 
    'floor_sf': {
        'values': [0]
    }, 
    'minhash_num_perm':
    {
        'values': [128]
    }, 
    'hll_p': {
        'values': [8]
    },
    'feature_prop': {
        'values': ['residual']
    },
    'use_feature': {
        'values': [True]

    },
    'propagate_embeddings': {
        'values': [False]
    },
    'sign_k': {
        'values': [0]
    },
    'use_zero_one': {
        'values': [0]
    }
    },

    'Wisconsin':{
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [128, 256]
    },
    'wd': {
        'values': [0]
    },
    'step': {
        'values': [0.1, 0.25, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'input_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'negatives': {
        'values': [0.25, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['gradient']
    }
    },
    'Cornell':{
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [128, 256]
    },
    'wd': {
        'values': [0]
    },
    'step': {
        'values': [0.1, 0.25, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'input_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'negatives': {
        'values': [0.25, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['hadamard']
    }
    }
}

parameters_dict_NCNC = { # 559872 possible confs
    'all_data':{
        'predictor': {
            'values': ["incn1cn1"]
        },
        'lr': {
            'values': [1e-2]
        },
        'hiddim': {
            'values': [128, 256]
        },
        'model': {
            'values': ['gcn', 'sage']
        },
        'wd': {
            'values': [0, 1e-3]
        },
        'mplayers': {
            'values': [1, 3, 5, 7, 12]
        },
        'nnlayers': {
            'values': [1, 2]
        },
        'predp': {
            'values': [0.3]
        },
        'gnndp': {
            'values': [0.3]
        },
        'negatives': {
            'values': [0.25, 0, 2, 4, 8]
        },
        'xdp': {
            'values': [0.4]
        },
        'tdp': {
            'values': [0.0]
        },
        'pt': {
            'values': [0.75]
        },
        'gnnedp': {
            'values': [0.0]
        },
        'preedp': {
            'values': [0.0]
        },
        'probscale': {
            'values': [6.5]
        },
        'proboffset': {
            'values': [4.4]
        },
        'alpha': {
            'values': [0.4]
        },
        'ln': {
            'values': [True]
        },
        'lnnn': {
            'values': [True]
        },
        'maskinput': {
            'values': [True]
        },
        'jk': {
            'values': [True]
        },
        'use_xlin': {
            'values': [True]
        },
        'res': {
            'values': [True]
        },
        'loadx': {
            'values': [False]
        },
        'readout_type': {
            'values': ['hadamard']
        },
        'lambda_': {
            'values': [True]
        },
        'depth': {
            'values': [1]
        },
        'splitsize': {
            'values': [-1]
        }


    }
}

parameters_dict_GRAFF = { # 559872 possible confs
    'all_data':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0]
    },
    'step': {
        'values': [0.1, 0.5]
    },
    'num_layers': {
        'values': [1, 3, 5, 7]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0, 0.1]
    },
    'input_dropout': {
        'values': [0.4, 0.5]
    },
    'lambda_': {
        'values': [0, 1e-5, 1e-10]
    },
    'negatives': {
        'values': [0.25, 0.5, 6, 8]
    },
    'normalize': {
        'values': [True]
    },
    'readout_type': {
        'values': ['gradient']
    }
    },
    'Wisconsin':{
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [128, 256]
    },
    'wd': {
        'values': [0]
    },
    'step': {
        'values': [0.1, 0.25, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'input_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'negatives': {
        'values': [0.25, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['gradient']
    }
    },
    'Cornell':{
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [128, 256]
    },
    'wd': {
        'values': [0]
    },
    'step': {
        'values': [0.1, 0.25, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'input_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'negatives': {
        'values': [0.25, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['hadamard']
    }
    },
    'tolokers':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-3]
    },
    'step': {
        'values': [0.5]
    },
    'num_layers': {
        'values': [1, 3, 5, 7, 12, 15]
    },
    'output_dim': {
        'values': [64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0.3]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [0.25, 0, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['hadamard']
    },
    'lambda_': {
        'values': [0]
    }
    },
    'Texas':{
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [128, 256]
    },
    'wd': {
        'values': [0]
    },
    'step': {
        'values': [0.1, 0.25, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'input_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'negatives': {
        'values': [0.25, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['gradient']
    }
    }
}


parameters_dict_SAGE = { # 174960 possible confs
    'all_data':{
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7]
    },
    'aggregation': {
        'values':['max', 'mean']
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }
    },'questions':{
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32, 64, 128]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7]
    },
    'aggregation': {
        'values':['max', 'mean']
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }
    },
    'Cornell':{
    'lr': {
        'values': [1e-1, 1e-2]
    },
    'hidden_dim': {
        'values': [256]
    },
    'wd': {
        'values': [0, 1e-6]
    },
    'num_layers': {
        'values': [2]
    },
    'aggregation': {
        'values':['mean']
    },
    'output_dim': {
        'values': [16]
    },
    'mlp_layer': {
        'values': [2]
    },
    'dropout': {
        'values': [0.6]
    },
    'input_dropout': {
        'values': [0.6]
    },
    'negatives': {
        'values': [6, 8, 10]
    }
    },
    'Wisconsin':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [256]
    },
    'wd': {
        'values': [0, 1e-6]
    },
    'num_layers': {
        'values': [3]
    },
    'aggregation': {
        'values':['mean']
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0]
    },
    'input_dropout': {
        'values': [0.6]
    },
    'negatives': {
        'values': [6, 8, 10]
    }
    },
    'tolokers':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-3]
    },
    'num_layers': {
        'values': [1, 3, 5, 7]
    },
    'output_dim': {
        'values': [64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0.3]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [0.25, 0, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['hadamard']
    },
    'lambda_': {
        'values': [0]
    }, 
    "res": {
        'values': [True]
    },
    "aggregation": {
        'values': ['mean', 'max']
    }
    },
    'Texas':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [32]
    },
    'wd': {
        'values': [0, 1e-6]
    },
    'num_layers': {
        'values': [7]
    },
    'aggregation': {
        'values':['max']
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [0]
    },
    'dropout': {
        'values': [0]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [6, 8, 10]
    }
    }
}

parameters_dict_GAT = { # 262440 possible confs
    'all_data': {
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256]
    },
    'wd': {
        'values': [0]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7]
    },
    'heads': {
        'values':[1]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }}
    ,
    'tolokers':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-3]
    },
    'num_layers': {
        'values': [1, 3, 5, 7]
    },
    'output_dim': {
        'values': [64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0.3]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [0.25, 0, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['hadamard']
    },
    'lambda_': {
        'values': [0]
    }, 
    "res": {
        'values': [True]
    },
    "heads": {
        'values': [1, 2]
    }
    },
    'questions': {
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [32, 64, 128]
    },
    'wd': {
        'values': [0]
    },
    'num_layers': {
        'values': [1, 2, 3, 5]
    },
    'heads': {
        'values':[1, 2, 4]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }
    }
}


parameters_dict_GCN = { # 87480 possible confs
    'all_data': {
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }
    },
    'amazon_ratings': { # 18 confs
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32]
    },
    'wd': {
        'values': [0]
    },
    'num_layers': {
        'values': [5, 7, 9]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [1, 2]
    },
    'dropout': {
        'values': [0.6]
    },
    'input_dropout': {
        'values': [0]
    },
    'negatives': {
        'values': [2]
    }
    },
    'tolokers':{
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-3]
    },
    'step': {
        'values': [1]
    },
    'num_layers': {
        'values': [1, 3, 5, 7, 12, 15]
    },
    'output_dim': {
        'values': [64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0.3]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [0.25, 0, 4, 8]
    },
    'normalize': {
        'values': [True]
    }, 
    'readout_type': {
        'values': ['hadamard']
    },
    'lambda_': {
        'values': [0]
    }, 
    "res": {
        'values': [True]
    }}
    ,
    'Wisconsin': { # 12 confs
    'lr': {
        'values': [1e-3]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [2, 3, 5, 7]
    },
    'output_dim': {
        'values': [16]
    },
    'mlp_layer': {
        'values': [1]
    },
    'dropout': {
        'values': [0]
    },
    'input_dropout': {
        'values': [0.6]
    },
    'negatives': {
        'values': [8]
    }
    }, 
    'Cornell': { # 12 confs
    'lr': {
        'values': [1e-3]
    },
    'hidden_dim': {
        'values': [32]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [2, 3, 5, 7]
    },
    'output_dim': {
        'values': [16]
    },
    'mlp_layer': {
        'values': [1]
    },
    'dropout': {
        'values': [0.6]
    },
    'input_dropout': {
        'values': [0]
    },
    'negatives': {
        'values': [2]
    }
    }, 
    'Texas': { # 12 confs
    'lr': {
        'values': [1e-3]
    },
    'hidden_dim': {
        'values': [32]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [2, 3, 5, 7]
    },
    'output_dim': {
        'values': [32]
    },
    'mlp_layer': {
        'values': [1]
    },
    'dropout': {
        'values': [0.3]
    },
    'input_dropout': {
        'values': [0.3]
    },
    'negatives': {
        'values': [8]
    }
    },
    'minesweeper': {
    'lr': {
        'values': [1e-1, 1e-2]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }
    },
    'questions': {
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32, 64, 128]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.3, 0.6]
    },
    'input_dropout': {
        'values': [0, 0.3, 0.6]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8]
    }
    }
}

parameters_dict_disenlink = { # 2592 possible confs
    'all_data': {
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'negatives': {
        'values': [0.25, 0.5, 0, 2, 4, 8],
    
    },
    'nfactor': {
        'values': [3, 6, 8, 12]
    
    }
    }
}
