import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from utils import *


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class EEG_DD(nn.Module):
    def __init__(self, num_feats):
        super(EEG_DD, self).__init__()
        self.linear1 = nn.Linear(num_feats, 64)
        self.linear2 = nn.Linear(64, 1)

    def GRL(self, x):
        return GradReverse.apply(x)

    def forward(self, embed):
        # input: (batch, 128)
        grl_out = self.GRL(embed)
        # (batch, 128)
        out = F.relu(self.linear1(grl_out))
        # (batch, 64)
        out = torch.sigmoid(self.linear2(out))
        # (batch, 1)
        return out.squeeze()


class EEG_LSTM(nn.Module):
    def __init__(self, num_feats, hidden):

        super(EEG_LSTM, self).__init__()
        self.hidden = hidden

        self.bnorm = nn.BatchNorm1d(32)
        self.lstm1 = nn.LSTM(num_feats, self.hidden[0], num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden[0], self.hidden[1], num_layers=1, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.3)

        self.linr1 = nn.Linear(self.hidden[1] * 32, 128)
        self.linr2 = nn.Linear(128, 1)
        self.atten = nn.Linear(self.hidden[1] * 32, 1)
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
    def __init__(self, eeg_path, dim, pretrained):

        super(CROSS_NN, self).__init__()
        self.eeg_net = self.load(eeg_path, pretrained)
        self.mus_net = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU(1e-2, inplace=True), nn.Dropout(0.5))
        self.drop = nn.Dropout(0.5)
        self.common = nn.Linear(128, dim)
        self.ddspace = EEG_DD(num_feats=dim)
        self.labels = nn.Linear(dim, 1)
        self.relu = nn.LeakyReLU(1e-2, inplace=True)

    def load(self, path, pretrained):
        m = (
            torch.load(path, map_location=device)
            if pretrained
            else EEG_LSTM(12, [128, 256]).to(device)
        )
        for prm in m.parameters():
            prm.requires_grad = True
        m.linr2 = nn.Sequential()
        return m

    def forward(self, eeg, mus):

        out1 = self.eeg_net(eeg.float())
        out2 = self.mus_net(mus.float())
        feat1 = self.relu(self.common(out1))
        feat2 = self.relu(self.common(out2))

        # domain discrimination
        perm = torch.randperm(feat1.shape[0])
        sample = perm[: feat1.shape[0] // 2]
        dd1, dd2 = feat1[sample], feat2[sample]
        dd_input = torch.cat((dd1, dd2)).to(device)
        dd_out = self.ddspace(dd_input[perm])
        labels = torch.cat((torch.ones(16), torch.zeros(16)))
        labels = labels[perm].to(device)

        pred1 = self.labels(self.drop(feat1)).squeeze()
        pred2 = self.labels(self.drop(feat2)).squeeze()
        pred1, pred2 = torch.sigmoid(pred1), torch.sigmoid(pred2)

        return feat1, feat2, pred1, pred2, dd_out, labels
