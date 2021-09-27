import numpy as np, os, torch

from utils import *
from preprocess import *
from loaders import get_loader
from models import *
from train import pretrain_model, cotrain_model
from test import test_on, test_participant

dur, subj = 3, range(12,13)
all_results = np.zeros(4)

if __name__ == '__main__':

    print('Program started. Utilized device: {}'.format(device))
    
    #process_DEAP_stimuli(num_tracks=34, eeg_dur=dur)
    #for i in range(32): process_DEAP_DE(datapath, str(i+1), dur=dur, exclude=True)
    
    for i in subj:
        
        p_dir = datapath + 'nets/nets_p{}/'.format(i)
        os.makedirs(p_dir, exist_ok=True)
        '''
        # pretrain using leave-one-subject-out cross validation
        print('\nPre-Training to test Participant {}.'.format(i+1))
        ind_loader = get_loader(datapath, mode='eeg', batch_size=128, dur=dur, subject=i)
        loader = {'train': ind_loader['train'], 'test': ind_loader['valid']}
        emodel = EEG_LSTM(num_feats=4*dur, hidden=[128,256]).to(device)
        optimizer = torch.optim.Adam(list(emodel.parameters()), lr=1e-3)
        emodel, _ = pretrain_model(emodel, loader, optimizer, patience=10, verbose=True)
        torch.save(emodel, p_dir+'EEG_net.pt')
        test_on(ind_loader['test'], emodel)
        '''
        # cotrain using 5-fold cross validation
        print('\nFine-tuning on Participant {}.'.format(i))
        crs_loader = get_loader(datapath, mode='cross', batch_size=32, dur=dur, subject=i)
        for fold in range(0):
                loader = {'train': crs_loader['train{}'.format(fold)], 'test': crs_loader['test{}'.format(fold)]}
                model = CROSS_NN(p_dir+'EEG_net.pt', dim=32).to(device)
                optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
                model, _ = cotrain_model(model, loader, optimizer, scheduler=None, patience=20, verbose=False)
                torch.save(model, p_dir+'CROSS_{}.pt'.format(fold))
        results = test_participant(p_dir, crs_loader, aggregate=True)
        all_results += [score/len(subj) for score in results]

    print("\n------------------------------------------------------------")
    print("Overall Stimulus Retrieval from EEG Queries (mAP): {:.2f} %".format(all_results[0]))
    print("Overall Related Track Retrieval from EEG Queries (mAP): {:.2f} %".format(all_results[1]))
    print("Overall EEG Emotion Classification (acc): {:.2f} %".format(all_results[2]))
    print("Overall MUS Emotion Classification (acc): {:.2f} %".format(all_results[3]))
