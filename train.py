import numpy as np, math, pickle, os, time, copy
import torch, torch.nn as nn, torchvision
from tqdm import tqdm

from utils import *

def calc_multiloss(view1_feat, view2_feat, view1_pred, view2_pred, labels_1, labels_2):
    
    siz = len(view1_feat)
    labels_1 = label_encoder(labels_1.cpu()).to(device).squeeze().float()
    labels_2 = label_encoder(labels_2.cpu()).to(device).squeeze().float()
    
    # a negative music pair for each EEG embedding
    dsim_feat = torch.zeros_like(view2_feat)
    for i in range(siz):
        match = np.random.randint(siz)
        if (1-labels_1[i,0]) in labels_2[:,0]:
            while labels_1[i,0] == labels_2[match,0]: match = np.random.randint(siz)
        dsim_feat[i] = view2_feat[match]
    
    # a negative EEG pair for each EEG embedding
    dsam_feat = torch.zeros_like(view1_feat)
    for i in range(siz):
        match = np.random.randint(siz)
        if (1-labels_1[i,0]) in labels_1[:,0]:
            while labels_1[i,0] == labels_1[match,0]: match = np.random.randint(siz)
        dsam_feat[i] = view1_feat[match]
    #print(view1_pred, view2_pred, view1_feat, view2_feat, labels_1, labels_2)
    # CE prediction loss
    pred_loss = nn.BCELoss()
    pred_loss1 = pred_loss(view1_pred, labels_1[:,0])
    pred_loss2 = pred_loss(view2_pred, labels_2[:,0])
    term1 = 0.7*pred_loss1 + 0.3*pred_loss2
    '''
    # HGR/CCA similarity loss
    #term2  = HGR(view1_feat, view2_feat, 1)
    #term2_loss = cca_loss(len(view1_feat), use_all_singular_values=True, device=device).loss
    #term2 = term2_loss(view1_feat,view2_feat)
    '''
    # triplet distance loss
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function = nn.CosineSimilarity())
    term21 = triplet_loss(view1_feat, view2_feat, dsam_feat)
    term22 = triplet_loss(view1_feat, view2_feat, dsim_feat)
    term2 = 0.5*term21 + 0.5*term22
    
    # invariance loss
    term3 = ((view1_feat-view2_feat)**2).sum(1).sqrt().mean()
    
    return 0.8*term1 + 0.2*term2

def calc_loss(out, labels):

	labels = label_encoder(labels.cpu()).to(device).squeeze().float()
	num = len(labels)
	loss = nn.BCELoss()
	loss = loss(out,labels[:,0])
	acc = torch.where(1*(out>0.5) == labels[:,0], torch.ones(num).to(device), torch.zeros(num).to(device)).sum()/len(out)
	return loss, acc

def pretrain_model(model, data_loader, optimizer, patience=5, num_epochs=50):

    begin = time.time()
    epoch_loss_history, rem  = [], patience
    best_model_wts, best = copy.deepcopy(model.state_dict()), 100.0

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 12)

        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            for samples, labels in tqdm(data_loader[phase], ncols=80):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    samples, labels = samples.to(device), labels[:,:2].to(device)
                    optimizer.zero_grad()
                    out = model(samples)
                    loss = calc_loss(out,labels)
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
                print('Test Loss: {:.5f}        acc: {:.4f}'.format(epoch_loss, acc))

                if epoch_loss < best:
                    best, rem = epoch_loss, patience
                    best_model_wts = copy.deepcopy(model.state_dict())
                else: rem -= 1
                epoch_loss_history.append(epoch_loss)

            else:  print('Train Loss: {:.5f}'.format(epoch_loss))
        if not rem: break

    time_elapsed = time.time() - begin
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_history

def cotrain_model(model, data_loader, optimizer, patience=5, num_epochs=100):

    begin = time.time()
    epoch_loss_history, rem = [], patience
    best_model_wts, best_acc = copy.deepcopy(model.state_dict()), 100.0

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 12)

        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            for eeg, mus in tqdm(data_loader[phase], ncols=80):
                eeg, elabel = eeg[0].to(device), eeg[1].to(device)
                mus, mlabel = mus[0].to(device), mus[1].to(device)
                optimizer.zero_grad()
                #print(eeg.shape, mus.shape, elabel.shape, mlabel.shape)
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    view1_feat, view2_feat, view1_pred, view2_pred = model(eeg, mus)
                    loss, retr = calc_loss(view1_pred, elabel)
                    loss = calc_multiloss(view1_feat, view2_feat, view1_pred, view2_pred, elabel, mlabel)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    #else: scheduler.step(loss.item())
                    running_loss += loss.item()
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            
            if phase == 'test':
                #scheduler.step(epoch_loss)
                #feats1, feats2, preds1, preds2, labels1, labels2 = generate_cross_preds(model, data_loader[phase])
                #labels1, labels2 = label_encoder(labels1).numpy(), label_encoder(labels2).numpy()
                #retr = retrieve(feats1, feats2, labels1, labels2, k=350, method='emotion')
                #retr = 0
                print('Test  Loss: {:.4f} \tEEG->MUS: {:.2f} %'.format(epoch_loss, retr))

                if epoch_loss < best_acc:
                    best_acc, rem, acc = epoch_loss, patience, retr
                    best_model_wts = copy.deepcopy(model.state_dict())
                else: rem -= 1
                epoch_loss_history.append(epoch_loss)

            else: print('Train Loss: {:.4f}'.format(epoch_loss))

        if not rem: break

    time_elapsed = time.time() - begin
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model, acc#epoch_loss_history

