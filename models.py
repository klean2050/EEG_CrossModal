import torch, torch.nn as nn
from utils import *

class EEG_LSTM(nn.Module):

	def __init__(self, num_feats, hidden):
		super(EEG_LSTM, self).__init__()
		self.hidden = hidden
		self.bnorm = nn.BatchNorm1d(32)
		self.lstm1 = nn.LSTM(num_feats, self.hidden[0], num_layers=1, batch_first=True)
		self.lstm2 = nn.LSTM(self.hidden[0], self.hidden[1], num_layers=1, batch_first=True)
		self.drop1 = nn.Dropout(0.2)
		self.drop2 = nn.Dropout(0.3)
		self.linr1 = nn.Linear(self.hidden[1]*32,128)
		self.linr2 = nn.Linear(128,1)
		self.atten = nn.Linear(self.hidden[1]*32,1)
		self.softm = nn.Softmax(dim=1)

	def forward(self, x):
		# normalization
		out = self.bnorm(x.float())
		# recurrent module
		out, _ = self.lstm1(out)
		out = self.drop1(out)
		out, _ = self.lstm2(out)
		out = self.drop1(out)
		# attention module
		out = out.flatten(start_dim=1)
		a = self.softm(self.atten(out))
		out = a.mul(out)
		# output embeddings
		out = self.linr1(out)
		out = self.drop2(out)
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
        m = torch.load(model, map_location=device)
        for prm in m.parameters(): prm.requires_grad = True
        m.linr2 = nn.Sequential()
        return m

    def forward(self, eeg, mus):
        out1, out2 = self.eeg_net(eeg.float()), self.mus_net(mus.float())
        view1_feat = self.relu(self.common(out1))
        view2_feat = self.relu(self.common(out2))
        view1_pred = self.labels(self.drop(view1_feat)).squeeze()
        view2_pred = self.labels(self.drop(view2_feat)).squeeze()
        view1_pred, view2_pred = torch.sigmoid(view1_pred), torch.sigmoid(view2_pred)
        return view1_feat, view2_feat, view1_pred, view2_pred

