import numpy as np, math, pickle, os, time, copy
import torch, torch.nn as nn, torchvision
from sklearn.model_selection import train_test_split
from scipy.signal import stft, butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import *


class EEG_LSTM(nn.Module):

	def __init__(self, num_feats, hidden):
		super(EEG_LSTM, self).__init__()
		self.hidden = hidden
		self.lstm1 = nn.LSTM(num_feats, self.hidden[0], num_layers=1, batch_first=True)
		self.lstm2 = nn.LSTM(num_feats, self.hidden[1], num_layers=1, batch_first=True)
		self.drop1 = nn.Dropout(0.3)
		self.drop2 = nn.Dropout(0.3)
		self.linr1 = nn.Linear(self.hidden[1]*32,128)
		self.linr2 = nn.Linear(128,1)

	def forward(self, x):
		x = x.float().squeeze(dim=1)
		out, _ = self.lstm1(x)
		out = self.drop1(out)
		out, _ = self.lstm2(x)
		out = out.flatten(start_dim=1)
		out = self.linr1(self.drop2(out))
		out = self.linr2(out)
		return torch.sigmoid(out).squeeze()

class CROSS_NN(nn.Module):

    def __init__(self, eeg_model, dim):
        super(CROSS_NN, self).__init__()
        self.eeg_net = self.load(eeg_model)
        self.mus_net = nn.Sequential(nn.Linear(128,128),
                                     nn.LeakyReLU(1e-2, inplace=True),
                                     nn.Dropout(0.5))
        self.drop = nn.Dropout(0.5)
        self.common  = nn.Linear(128,dim)
        self.labels  = nn.Linear(dim,1)
        self.relu = nn.LeakyReLU(1e-2, inplace=True)

    def load(self, model):
        m = torch.load(datapath+'nets/{}.pt'.format(model), map_location='cpu')
        for prm in m.parameters(): prm.requires_grad = True
        m.linr2 = nn.Sequential()
        return m

    def forward(self, eeg, mus):
        out1, out2 = self.eeg_net(eeg), self.mus_net(mus)
        view1_feat = self.relu(self.common(out1))
        view2_feat = self.relu(self.common(out2))
        view1_pred = self.labels(self.drop(view1_feat)).squeeze()
        view2_pred = self.labels(self.drop(view2_feat)).squeeze()
        view1_pred, view2_pred = torch.sigmoid(view1_pred), torch.sigmoid(view2_pred)
        return view1_feat, view2_feat, view1_pred, view2_pred

