import numpy as np, math, pickle, os, time, copy
import torch, torch.nn as nn, torchvision
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from scipy.signal import stft, butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import *
from preprocess import process_DEAP_DE


class SINGLEDataSet(Dataset):
        ''' Dataset for the pre-train task '''
        def __init__(self, modal, labels):
                self.data = modal
                self.labels = labels

        def __getitem__(self, index):
                return (self.data[index], self.labels[index])

        def __len__(self):
                return len(self.data)

class CROSSDataSet(Dataset):
    ''' Dataset for the cross-modal task '''
    def __init__(self, eeg, mus, elabels, mlabels):
        self.eeg = {'data': eeg, 'label': elabels}
        self.mus = {'data': mus, 'label': mlabels}

    def __getitem__(self, index):
        eeg = (self.eeg['data'][index], self.eeg['label'][index])
        mus = (self.mus['data'][index], self.mus['label'][index])
        return eeg, mus

    def __len__(self):
        return len(self.eeg['data'])

def get_de_loader(path, dur, subject, num_tracks=34):

        num = (60 - dur + 1) * num_tracks
        eeg_samples = np.zeros((32,num,32,4*dur))
        eeg_labels = np.zeros((32,num,5))
        for s in range(32):
                c = '0'+str(s+1) if s<9 else str(s+1)
                eeg_samples[s] = np.load(path+'{}sec_de/P{}_feats.npy'.format(dur,c))
                eeg_labels[s]  = np.load(path+'{}sec_de/P{}_annot.npy'.format(dur,c))

        lst = list(range(32)); lst.remove(subject)
        X_train = eeg_samples[lst]; X_test = eeg_samples[subject]
        y_train = eeg_labels[lst];  y_test = eeg_labels[subject]

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=2)

        eeg, lab = {}, {}
        eeg['train'] = X_train.reshape(27*num,32,4*dur)
        eeg['valid'] = X_valid.reshape(4*num,32,4*dur)
        eeg['test']  = X_test.reshape(num,32,4*dur)
        lab['train'] = y_train.reshape(27*num,-1)
        lab['valid'] = y_valid.reshape(4*num,-1)
        lab['test']  = y_test.reshape(num,-1)

        return eeg, lab

def get_crs_loader(path, dur, subject):

        c = '0'+str(subject+1) if subject<9 else str(subject+1)
        eeg_samples = np.load(path+'{}sec_de/P{}_feats.npy'.format(dur,c)).reshape(34,58,32,-1)
        eeg_labels  = np.load(path+'{}sec_de/P{}_annot.npy'.format(dur,c)).reshape(34,58,-1)
        mus_samples = np.load(path+'stimuli/tracks_embeds.npy')
        mus_labels  = np.load(path+'stimuli/tracks_labels.npy')

        to_stratify = label_encoder(eeg_labels[:,0,:].squeeze())

        eeg, lab, mus, mlab = {}, {}, {}, {}
        s = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2)
        for fold, (train_index, test_index) in enumerate(s.split(eeg_samples, to_stratify[:,0])):
                
                X_train = eeg_samples[train_index]; X_test = eeg_samples[test_index]
                y_train = eeg_labels[train_index];  y_test = eeg_labels[test_index]

                eeg['train{}'.format(fold)] = X_train.reshape(X_train.shape[0]*58,32,4*dur)
                eeg['test{}'.format(fold)]  = X_test.reshape(X_test.shape[0]*58,32,4*dur)
                lab['train{}'.format(fold)] = y_train.reshape(y_train.shape[0]*58,-1)
                lab['test{}'.format(fold)]  = y_test.reshape(y_test.shape[0]*58,-1)

        for x in eeg.keys(): mus[x], mlab[x] = make_pairs(eeg[x], lab[x], mus_samples, mus_labels, dur)
        return eeg, mus, lab, mlab


def get_loader(path, mode, batch_size, dur, subject='0'):

        if mode == 'eeg':
                eeg, elabels = get_de_loader(path, dur, subject)
                dataset = {x: SINGLEDataSet(modal=eeg[x], labels=elabels[x]) for x in eeg.keys()}
        else:
                eeg, mus, elabels, mlabels = get_crs_loader(path, dur, subject)
                dataset = {x:CROSSDataSet(eeg=eeg[x], mus=mus[x], elabels=elabels[x], mlabels=mlabels[x]) for x in eeg.keys()}
        
        dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, num_workers=0, drop_last=True) for x in dataset.keys()}
        return dataloader

