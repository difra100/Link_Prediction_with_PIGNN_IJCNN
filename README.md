# Link Prediction with Physics-Inspired Graph Neural Networks

This repository contains the code and to train and reproduce the experiments in the paper 'Link Prediction with Physics-Inspired Graph Neural Networks'.   
We have available the appendix of the paper in the 'appendix.pdf' file. 
We propose GRAFF-LP a link prediction pipeline built upon PIrd GNNs and a novel readout function that enhances the performances.  
The general overview of our framework follows.  
![Example Image](architecture.png)  
Data for **amazon_ratings** and **questions** are available on request to the authors.    


### Setup and Installation of the Environment

```
conda create -n GRAFFLP python=3.9 && conda activate GRAFFLP && conda install pip
```
### Get the required libraries following these bash commands
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
pip install -r requirements.txt
```

### To print the significance test results
```
python significance_test.py
```
Now the code to reproduce the experiments follows. Note that by changing the argument `readout_type` from hadamard to gradient, we obtain the performance improvement specified in the paper.  

### Download our data splits!
Before trying to reproduce the experiments, you should populate the `data/` folder with the files downloaded at this [link](https://www.dropbox.com/scl/fi/b8g6pyyipldjfnwdjjmvw/link_pred_data.zip?rlkey=9jifqnti268dcnsuayelqq59u&st=hpxi88ye&dl=0).


## GRAFF hyperparameters for amazon_ratings
```
python main.py -dataset_name 'amazon_ratings' -model_name 'GRAFF' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 12 -step 0.3 -hidden_dim 128 -output_dim 32 -mlp_layer 1 -dropout 0.4 -input_dropout 0 -negatives 4 -readout_type 'hadamard'
```

## GRAFF hyperparameters for questions
```
python main.py -dataset_name 'questions' -model_name 'GRAFF' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0.001 -num_layers 7 -step 0.5 -hidden_dim 128 -output_dim 32 -mlp_layer 1 -dropout 0.6 -input_dropout 0 -negatives 4 -readout_type 'hadamard'
```

## GRAFF hyperparameters for minesweeper
```
python main.py -dataset_name 'minesweeper' -model_name 'GRAFF' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 9 -step 0.1 -hidden_dim 64 -output_dim 64 -mlp_layer 1 -dropout 0.6 -input_dropout 0.6 -negatives 2 -readout_type 'hadamard'
```

## GRAFF hyperparameters for roman_empire
```
python main.py -dataset_name 'roman_empire' -model_name 'GRAFF' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 1 -step 0.75 -hidden_dim 128 -output_dim 32 -mlp_layer 1 -dropout 0.4 -input_dropout 0 -negatives 4 -readout_type 'hadamard'
```

## GCN hyperparameters for amazon_ratings
```
python main.py -dataset_name 'amazon_ratings' -model_name 'GCN' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 5 -hidden_dim 32 -output_dim 32 -mlp_layer 1 -dropout 0.6 -input_dropout 0 -negatives 2 -readout_type 'hadamard'
```

## GCN hyperparameters for questions
```
python main.py -dataset_name 'questions' -model_name 'GCN' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 7 -hidden_dim 32 -output_dim 16 -mlp_layer 1 -dropout 0.6 -input_dropout 0.3 -negatives 2 -readout_type 'hadamard'
```

## GCN hyperparameters for minesweeper
```
python main.py -dataset_name 'minesweeper' -model_name 'GCN' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 3 -hidden_dim 128 -output_dim 32 -mlp_layer 2 -dropout 0.6 -input_dropout 0.6 -negatives 2 -readout_type 'hadamard'
```

## GCN hyperparameters for roman_empire
```
python main.py -dataset_name 'roman_empire' -model_name 'GCN' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0.01 -num_layers 1 -hidden_dim 128 -output_dim 32 -mlp_layer 2 -dropout 0.6 -input_dropout 0.6 -negatives 0 -readout_type 'hadamard'
```

## SAGE hyperparameters for amazon_ratings
```
python main.py -dataset_name 'amazon_ratings' -model_name 'SAGE' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0.001 -num_layers 1 -hidden_dim 256 -output_dim 64 -mlp_layer 2 -dropout 0 -input_dropout 0.6 -negatives 4 -aggregation 'mean' -readout_type 'hadamard'
```

## SAGE hyperparameters for questions
```
python main.py -dataset_name 'questions' -model_name 'SAGE' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 5 -hidden_dim 32 -output_dim 32 -mlp_layer 0 -dropout 0.6 -input_dropout 0 -negatives 0.5 -aggregation 'mean' -readout_type 'hadamard'
```
## SAGE hyperparameters for minesweeper
```
python main.py -dataset_name 'minesweeper' -model_name 'SAGE' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 5 -hidden_dim 256 -output_dim 16 -mlp_layer 0 -dropout 0 -input_dropout 0.6 -negatives 4 -aggregation 'mean' -readout_type 'hadamard'
```

## SAGE hyperparameters for roman_empire
```
python main.py -dataset_name 'roman_empire' -model_name 'SAGE' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 1 -hidden_dim 256 -output_dim 16 -mlp_layer 1 -dropout 0.3 -input_dropout 0.6 -negatives 0.25 -aggregation 'mean' -readout_type 'hadamard'
```

## GAT hyperparameters for amazon_ratings
```
python main.py -dataset_name 'amazon_ratings' -model_name 'GAT' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.001 -wd 0 -num_layers 3 -hidden_dim 256 -output_dim 64 -mlp_layer 0 -dropout 0 -input_dropout 0.3 -negatives 0.5 -heads 1 -readout_type 'hadamard'
```

## GAT hyperparameters for questions
```
python main.py -dataset_name 'questions' -model_name 'GAT' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 1 -hidden_dim 128 -output_dim 64 -mlp_layer 0 -dropout 0.6 -input_dropout 0.3 -negatives 2 -heads 1 -readout_type 'hadamard'
```


## GAT hyperparameters for minesweeper
```
python main.py -dataset_name 'minesweeper' -model_name 'GAT' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.001 -wd 0 -num_layers 7 -hidden_dim 64 -output_dim 64 -mlp_layer 2 -dropout 0.3 -input_dropout 0.3 -negatives 4 -heads 1 -readout_type 'hadamard'
```
## GAT hyperparameters for roman_empire
```
python main.py -dataset_name 'roman_empire' -model_name 'GAT' -device 'cpu' -eval True -num_data_splits 1 -mode 'test' -lr 0.01 -wd 0 -num_layers 1 -hidden_dim 256 -output_dim 16 -mlp_layer 2 -dropout 0 -input_dropout 0.6 -negatives 0.5 -heads 1 -readout_type 'hadamard'
```

## MLP hyperparameters for amazon_ratings
```
python main.py -dataset_name 'amazon_ratings' -num_data_splits 1 -model_name 'mlp' -mode 'test' -device 'cpu' -lr 0.001 -wd 0 -num_layers 1 -hidden_dim 128 -output_dim 64 -mlp_layer 0  -dropout 0.4 -input_dropout 0.5 -negatives 0 -eval True -save_performance True -readout_type 'hadamard'
```

## MLP hyperparameters for roman_empire
``` 
python main.py -dataset_name 'roman_empire' -num_data_splits 1 -model_name 'mlp' -mode 'test' -device 'cpu' -lr 0.01 -wd 0 -num_layers 1 -hidden_dim 128 -output_dim 64 -mlp_layer 0  -dropout 0 -input_dropout 0 -negatives 0.25 -eval True -save_performance True -readout_type 'hadamard'
```

## MLP hyperparameters for minesweeper
```
python main.py -dataset_name 'minesweeper' -num_data_splits 1 -model_name 'mlp' -mode 'test' -device 'cpu' -lr 0.01 -wd 0 -num_layers 3 -hidden_dim 128 -output_dim 64 -mlp_layer 0 -dropout 0.4 -input_dropout 0.5 -negatives 0 -eval True -save_performance True -readout_type 'hadamard'
```

## MLP hyperparameters for questions
```
python main.py -dataset_name 'questions' -num_data_splits 1 -model_name 'mlp' -mode 'test' -device 'cpu' -lr 0.01 -wd 0.001 -num_layers 1 -hidden_dim 128 -output_dim 64 -mlp_layer 0 -dropout 0.5 -input_dropout 0 -negatives 8 -eval True -save_performance True -readout_type 'hadamard'
```







