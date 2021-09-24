import numpy as np, math, pickle, os, time, copy
import torch, torch.nn as nn, torchvision
from sklearn.model_selection import train_test_split
from scipy.signal import stft, butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils import *
from preprocess import process_DEAP_DE
from loaders import get_loader
from models import *
from train import pretrain_model, cotrain_model
from test import test_on, test_fold

dur = 3

if __name__ == '__main__':

    print('Program started. Utilized device: {}'.format(device))
    #for i in range(32): process_DEAP_DE(datapath, str(i+1), dur=dur, exclude=True)
    
    for i in range(2):
        
        p_dir = datapath + 'nets/nets_p{}/'.format(i)
        os.makedirs(p_dir, exist_ok=True)
        '''
        # pretrain using leave-one-subject-out cross validation
        print('\nPre-Training to test Participant {}.'.format(i+1))
        ind_loader = get_loader(datapath, mode='eeg', batch_size=128, dur=dur, subject=i)
        loader = {'train': ind_loader['train'], 'test': ind_loader['valid']}
        emodel = EEG_LSTM(num_feats=4*dur, hidden=[128,256]).to(device)
        optimizer = torch.optim.Adam(list(emodel.parameters()), lr=1e-4)
        emodel, _ = pretrain_model(emodel, loader, optimizer, patience=10, verbose=True)
        torch.save(emodel, p_dir+'EEG_net.pt')
        test_on(ind_loader['test'], emodel)
        '''
        # cotrain using leave-one-trial-out cross validation for the test
        print('\nFine-tuning on Participant {}.'.format(i+1))
        for fold in range(5):
                crs_loader = get_loader(datapath, mode='cross', batch_size=32, dur=dur, subject=i)
                loader = {'train': crs_loader['train{}'.format(fold)], 'test': crs_loader['test{}'.format(fold)]}
                model = CROSS_NN(p_dir+'EEG_net.pt', dim=64).to(device)
                optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
                scheduler = ReduceLROnPlateau(optimizer, patience=6, verbose=True)
                model, _ = cotrain_model(model, loader, optimizer, scheduler, patience=15, verbose=True)
                torch.save(model, p_dir+'CROSS_{}.pt'.format(fold))
                test_fold(p_dir, fold, crs_loader['test{}'.format(fold)], aggregate=False)
