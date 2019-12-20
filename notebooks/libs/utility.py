import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def prepre_inputs(sce, assay, val_pro, seed, sort_gene = False, filter_threshold = None, hvg = True, transformer = None):
    """Take in a Annadata object, create X_train, X_val, X_test, y_train, y_val, y_test 
    
    Parameters
    ----------
    sce (Annadata): SingleCellExperiment -> Annadata 
    assay (str): either 'raw' or 'normalized'
    val_pro (float): proportion of val to extract from train, between [0. , 1.]
    seed (int): use to create train, val from train 
    transformer (sklearn transformer): fit using train before transform train, val, test
    
    Returns
    -------
    tuple: X_train, X_val, X_test, y_train, y_val, y_test 

    """
    
    if sort_gene: sce = sce[:, sce.uns['gene_order']]
    if filter_threshold is not None:
        gene_keep = np.where((sce[sce.obs['split'] == 'train', :].layers['logcounts'].todense().sum(0) > filter_threshold))[1]
        sce = sce[:, gene_keep]
    #if hvg: sce = sce[:, sce.uns['hvg_order'][:hvg]] 
    if hvg: sce = sce[:, sce.var['is_hvgs'] == 1]
        
    sce_train = sce[sce.obs['split'] == 'train', ]
    sce_test = sce[sce.obs['split'] == 'test', ]
    
    if assay == 'raw':
        X_train = sce_train.X.todense()
        X_test = sce_test.X.todense()
    elif assay == 'normalized':
        X_train = sce_train.layers['logcounts'].todense()
        X_test = sce_test.layers['logcounts'].todense()
    
    y_train = sce_train.obs['pos']
    y_test = sce_test.obs['pos']
    
    # split train into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_pro, random_state = seed, 
                                                      shuffle = True, stratify = y_train)
    if(transformer):
        transformer = transformer.fit(X_train)
        X_train = transformer.transform(X_train)
        X_val = transformer.transform(X_val)
        X_test = transformer.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test 

## we need a custom Dataset class to work with numpy and pandas objects
class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        if isinstance(target, pd.Series):
            target = target.to_numpy()
        self.target = torch.from_numpy(target).float() 
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data) # number of rows = # samples
    
def get_dataset(X, y, custom_dataset, transform = None):
    """Create custom torch Dataset using custom_dataset class
    
    Parameters
    ----------
    data: X, y
    custom_dataset (callable, a torch Dataset class)
    
    Returns
    -------
    dataset object
    
    """
    
    dataset = custom_dataset(X, y, transform)
    
    return dataset

def get_dataloader(dataset, shuffle, batch_size = None):
    """Create Dataloader from inputs dataset
    
    Parameters
    ----------
    dataset: dataset object
    batch_size (int): if None, batch size = total number fo samples
    
    Returns
    -------
    dataloader object
    
    """
    
    if(batch_size): # mini-batch GD
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else: # batch GD, default to this since entire set of samples fit memory
        dl = DataLoader(dataset, len(dataset), shuffle=shuffle)
    return dl    

# adapt from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.patience_counter = 0
        self.es_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model, checkpoint_dir = None):
        
        self.es_counter += 1

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if checkpoint_dir: self.save_checkpoint(val_loss, model, checkpoint_dir = None)
        elif score < self.best_score + self.delta:
            self.patience_counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.patience_counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if checkpoint_dir: self.save_checkpoint(val_loss, model, checkpoint_dir = None)
            self.best_model = model.state_dict()
            self.patience_counter = 0
            
    def save_checkpoint(self, val_loss, model, checkpoint_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'early_stopping_checkpoint.pt'))    
        self.val_loss_min = val_loss
    