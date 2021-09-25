import numpy as np, math, pickle, os, time, copy
import torch, torch.nn as nn, torchvision
from tqdm import tqdm

from test import get_cross_preds, retrieve
from utils import *

def calc_class_weights(labels, eps=.1):

    minority_class = 0 if sum(labels) > len(labels)/2 else 1
    minority_weight = len(labels) / sum(labels == minority_class)
    majority_weight = len(labels) / sum(labels != minority_class)
    
    normalized_min = minority_weight*0.8
    normalized_max = majority_weight

    batch_weights = [normalized_min if i == minority_class else normalized_max for i in labels]
    return torch.Tensor(batch_weights).to(device)

def calc_multiloss(view1_feat, view2_feat, view1_pred, view2_pred, labels_1, labels_2, phase):
    
    labels_1 = label_encoder(labels_1.cpu()).to(device).squeeze().float()[:,0]
    labels_2 = label_encoder(labels_2.cpu()).to(device).squeeze().float()
    
    batch_weights1 = calc_class_weights(labels_1) if phase == 'train' else None
    batch_weights2 = calc_class_weights(labels_2) if phase == 'train' else None
    # CE prediction loss
    pred_loss = nn.BCELoss(weight=batch_weights1)
    pred_loss1 = pred_loss(view1_pred, labels_1)
    pred_loss = nn.BCELoss(weight=batch_weights2)
    pred_loss2 = pred_loss(view2_pred, labels_2)
    term1 = 0.6*pred_loss1 + 0.4*pred_loss2
    
    # COS metric loss
    cosine_loss = nn.CosineEmbeddingLoss()
    term2 = cosine_loss(view1_feat, view2_feat, torch.ones(len(view1_feat)).to(device))
    
    return 0.8*term1 + 0.2*term2

def calc_loss(out, labels, phase):

	labels = label_encoder(labels.cpu()).to(device).squeeze().float()
	batch_weights = calc_class_weights(labels) if phase == 'train' else None
	loss = nn.BCELoss(weight=batch_weights)
	return loss(out,labels)

def pretrain_model(model, data_loader, optimizer, patience=5, num_epochs=50, verbose=False):

    begin = time.time()
    epoch_loss_history, rem  = [], patience
    best_model_wts, best = copy.deepcopy(model.state_dict()), 100.0
    vprint = print if verbose else lambda *a, **k: None

    for epoch in range(num_epochs):
        vprint('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        vprint('-' * 12)

        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            for samples, labels in tqdm(data_loader[phase], ncols=80, disable = not verbose):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    samples, labels = samples.to(device), labels[:,:2].to(device)
                    optimizer.zero_grad()
                    out = model(samples)
                    loss = calc_loss(out,labels,phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
            epoch_loss = running_loss / len(data_loader[phase].dataset)

            if phase == 'test':

                t_preds, t_labels = [], []
                with torch.no_grad():
                    for samples, labels in data_loader[phase]:
                        samples, labels = samples.to(device), labels[:,:2].to(device)
                        out = model(samples)
                        out, labels = 1*out>0.5, label_encoder(labels.cpu()).to(device).squeeze()
                        t_preds.append(out.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                t_preds  = np.concatenate(t_preds)
                t_labels = np.concatenate(t_labels).squeeze()

                acc = (np.mean(sum(t_preds==t_labels)) * 100.0 / len(t_labels))
                vprint('Test Loss: {:.5f}        acc: {:.4f}'.format(epoch_loss, acc))

                if epoch_loss < best:
                    best, rem = epoch_loss, patience
                    best_model_wts = copy.deepcopy(model.state_dict())
                else: rem -= 1
                epoch_loss_history.append(epoch_loss)

            else:  vprint('Train Loss: {:.5f}'.format(epoch_loss))
        if not rem: break

    time_elapsed = time.time() - begin
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_history

def cotrain_model(model, data_loader, optimizer, scheduler=False, patience=5, num_epochs=100, verbose=False):

    begin = time.time()
    epoch_loss_history, rem = [], patience
    best_model_wts, best_acc = copy.deepcopy(model.state_dict()), 100.0
    vprint = print if verbose else lambda *a, **k: None

    for epoch in range(num_epochs):
        vprint('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        vprint('-' * 12)

        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            for eeg, mus in tqdm(data_loader[phase], ncols=80, disable = not verbose):
                eeg, elabel = eeg[0].to(device), eeg[1].to(device)
                mus, mlabel = mus[0].to(device), mus[1].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    view1_feat, view2_feat, view1_pred, view2_pred = model(eeg, mus)
                    loss = calc_multiloss(view1_feat, view2_feat, view1_pred, view2_pred, elabel, mlabel, phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            
            if phase == 'test':
                if scheduler: scheduler.step(epoch_loss)
                feats1, feats2, preds1, preds2, labels1, labels2 = get_cross_preds(model, data_loader[phase])
                labels1, labels2 = label_encoder(labels1).numpy(), label_encoder(labels2).numpy()
                retr = retrieve(feats1, feats2, labels1, labels2, k=len(feats1), method='emotion')
                vprint('Test  Loss: {:.4f} \tEEG->MUS: {:.2f} %'.format(epoch_loss, 100*np.mean(retr)))

                if epoch_loss < best_acc:
                    best_acc, rem = epoch_loss, patience
                    best_model_wts = copy.deepcopy(model.state_dict())
                else: rem -= 1
                epoch_loss_history.append(epoch_loss)

            else: vprint('Train Loss: {:.4f}'.format(epoch_loss))

        if not rem: break

    time_elapsed = time.time() - begin
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_history

