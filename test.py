import numpy as np, math, pickle, os, time, copy
import torch, torch.nn as nn, torchvision
from sklearn.model_selection import train_test_split
from scipy.signal import stft, butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import *


def test_on(test_loader, model):

        t_preds, t_labels = [], []
        with torch.no_grad():
                for samples, labels in test_loader:
                        samples, labels = samples.to(device), labels[:,:2].to(device)
                        out = model(samples)
                        out, labels = 1*out>0.5, label_encoder(labels.cpu()).to(device).squeeze()
                        t_preds.append(out.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
        t_preds  = np.concatenate(t_preds)
        t_labels = np.concatenate(t_labels).squeeze()

        acc = (np.mean(sum(t_preds==t_labels)) * 100.0 / len(t_labels))
        print('Test accuracy: {:.4f}'.format(acc))

def get_cross_preds(model, loader):

	t_eeg, t_mus, p_eeg, p_mus, e_lab, m_lab = [], [], [], [], [], []
	with torch.no_grad():
		for eeg, mus in loader:
			t_view1_feat, t_view2_feat, t_view1_pred, t_view2_pred = model(eeg[0].to(device), mus[0].to(device))
			t_eeg.append(t_view1_feat.cpu().numpy())
			t_mus.append(t_view2_feat.cpu().numpy())
			p_eeg.append(t_view1_pred.cpu().numpy())
			p_mus.append(t_view2_pred.cpu().numpy())
			e_lab.append(eeg[1].cpu().numpy())
			m_lab.append(mus[1].cpu().numpy())
		t_eeg = np.concatenate(t_eeg)
		t_mus = np.concatenate(t_mus)
		p_eeg = np.concatenate(p_eeg)
		p_mus = np.concatenate(p_mus)
		e_lab = np.concatenate(e_lab)
		m_lab = np.concatenate(m_lab)
	return t_eeg, t_mus, p_eeg, p_mus, e_lab, m_lab

def retrieve(eeg, mus, elabels, mlabels, k=0, method='track'):
    
    dist = cdist(eeg, mus, 'cosine')
    sort, res = dist.argsort(), []
    for i in range(dist.shape[0]):
        order, r, p = sort[i], 0.0, 0.0
        for j in range(k):
            if (method == 'emotion' and elabels[i,0] == mlabels[order[j],0]) \
            or (method == 'track'   and elabels[i,1] == mlabels[order[j],-1]):
                r += 1.0
                p += r/(j+1)
        res += [p/r] if r else [0] # mAP
    return res

def test_trial(p, trial, loader, aggregate=False):
	
	model_name = datapath + 'nets/CROSS_{}_{}.pt'.format(p,trial)
	model = torch.load(model_name).to(device)
	model.eval()

	feats1, feats2, preds1, preds2, labels1, labels2 = get_cross_preds(model, loader)
	labels1 = label_encoder(labels1).numpy()
	labels2 = label_encoder(labels2).numpy()
		
	#print("Exporting t-SNE of the common latent space")
	#features = np.concatenate((feats1, feats2))
	#labels = np.concatenate((labels1[:,0], labels2[:,0]))
	#visualize(features, labels, name=m)
		
	retr_track = retrieve(feats1, feats2, labels1, labels2, k=350, method='track')
	if not aggregate: retr_track = 100*np.mean(retr_track)
	else:  retr_track = 100*test_tracks_retr(retr_track, labels1)
		
	retr_emot = retrieve(feats1, feats2, labels1, labels2, k=350, method='emotion')
	if not aggregate: retr_emot = 100*np.mean(retr_emot)
	else:  retr_emot = 100*test_tracks_retr(retr_emot, labels1)
		
	#print(confusion_matrix(labels1[:,0], preds1.argmax(axis=1)))
		
	if not aggregate: pred_emot = 100*sum(preds1.argmax(axis=1)==labels1[:,0])/350
	else:  pred_emot = test_tracks(preds1.argmax(axis=1), labels1)
		
	if not aggregate: pred_mus = 100*sum(preds2.argmax(axis=1)==labels2[:,0])/350
	else:  pred_mus = test_music(preds2.argmax(axis=1), labels2)
	
	print("Stimulus Retrieval from EEG Queries (mAP): {:.2f} %".format(retr_track))
	print("Related Track Retrieval from EEG Queries (mAP): {:.2f} %".format(retr_emot))
	print("EEG Emotion Classification (acc): {:.2f} %".format(pred_emot))
	print("MUS Emotion Classification (acc): {:.2f} %".format(pred_mus))
