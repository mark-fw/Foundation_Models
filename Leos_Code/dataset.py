import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

#creates a torch dataset from a preprocessed h5 file
class JetDataSet(Dataset):
    def __init__(self, data_dir, tag : str, 
                 num_features=3,
                 num_bins = (40, 30, 30),
                 num_const = 50,
                 num_jets = False,
                 add_stop = True,
                 add_start = True
                 ):
        if num_jets == False:
            df = pd.read_hdf(data_dir, key = "df")
        else:
            df = pd.read_hdf(data_dir, key = "df").head(num_jets) 
        self.data = disc_to_token(df,
                                  num_features=num_features,
                                  num_bins=num_bins,
                                  num_const=num_const,
                                  add_end=add_stop,
                                  add_start=add_start)
        self.tag = tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#takes in binned discretized data as a dataframe and outputs tokens as torch tensor that can be used for training/validating or other purposes
def disc_to_token(df, 
                  num_features, #number of different features(pt, eta, phi)
                  num_bins,
                  num_const, #number of constituents per jet, this can be limited
                  to_tensor = True, #if we want to return a torch tensor
                  add_start = False, #wether to add start and end tokens
                  add_end = False, 
                  ): 
    
    x = df.to_numpy(dtype = np.int64)[:, : num_const * num_features] # this keeps only as many constituents we want in our data
    x = x.reshape(x.shape[0], -1, num_features) #this reshapes the data such that its 3 dimensional with [njets, nconst, nfeatures] [[[pt_1, eta_1, phi_1], ...],[[pt_1, eta_1, phi_1], ...],...]
    
    x = x.copy()

    padding_mask = x == -1 #marks every where a invalid const is

    #add start and stop token if needed
    if add_start: 
        #this shifts every valid bin --> 1 so the 0 can now be the start token
        x[~padding_mask] += 1
        
        #this adds a start particle with (0,0,0) to the start of every jet
        x = np.concatenate(
            (
                np.zeros((len(x), 1, num_features), dtype=int),
                x,
            ),
            axis=1,
        )

        num_bins= [x +1 for x in num_bins]
        print("Added start token. New bins are now:", num_bins)
    #add stop token only if the actual number of const. in the jet is smaller than the limit we have set for const.
    #so if a jet fills all the const dont set a stop token
    if add_end:
        num_bins= [x +1 for x in num_bins]
        
        #compute length of each jet
        jet_length = (~padding_mask[:, :, 0]).sum(1) + 1 #this gives the index of the first invalid const. +1 because of start token
        valid = (jet_length >= 0) & (jet_length < x.shape[1]) #this ensures that the index we want to set to the stop token is not out of bounds 

        x[np.arange(x.shape[0])[valid], jet_length[valid]] = num_bins        

        print("Added stop token. New bins are now:", num_bins)

    if to_tensor: 
        x = torch.tensor(x)

    return x

    