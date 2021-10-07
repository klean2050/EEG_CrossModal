import numpy as np, time, copy
import torch, torch.nn as nn
from tqdm import tqdm

from eeg_cmr.test import get_cross_preds, retrieve
from eeg_cmr.utils import *


def calc_class_weights(labels, eps=0.75):

    minority_class = 0 if sum(labels) > len(labels) / 2 else 1
    minority_weight = len(labels) / sum(labels == minority_class)
    majority_weight = len(labels) / sum(labels != minority_class)

    normalized_min = minority_weight * eps
    normalized_max = majority_weight

    batch_weights = [normalized_min if i == minority_class else normalized_max for i in labels]
    return torch.Tensor(batch_weights).to(device)


def calc_multiloss(pred1, pred2, dd_out, label1, label2, dd_label, phase):

    label1 = label_encoder(label1.cpu()).to(device).squeeze().float()[:, 0]
    label2 = label_encoder(label2.cpu()).to(device).squeeze().float()[:, 0]

    batch_weights1 = calc_class_weights(label1) if phase == "train" else None
    batch_weights2 = calc_class_weights(label2) if phase == "train" else None

    # CE prediction loss
    pred_loss = nn.BCELoss(weight=batch_weights1)
    pred_loss1 = pred_loss(pred1, label1)
    pred_loss = nn.BCELoss(weight=batch_weights2)
    pred_loss2 = pred_loss(pred2, label2)
    term1 = 0.6 * pred_loss1 + 0.4 * pred_loss2

    # Domain discrimination loss
    dd_loss = nn.BCELoss()
    term2 = dd_loss(dd_out, dd_label)

    return 0.9 * term1 + 0.1 * term2


def calc_loss(out, labels, phase):

    labels = label_encoder(labels.cpu()).to(device).squeeze().float()
    batch_weights = calc_class_weights(labels) if phase == "train" else None
    loss = nn.BCELoss(weight=batch_weights)
    return loss(out, labels)


def pretrain_model(model, data_loader, optimizer, patience, num_epochs=50, verbose=False):

    begin = time.time()
    epoch_loss_history, rem = [], patience
    best_model_wts, best = copy.deepcopy(model.state_dict()), 100.0
    vprint = print if verbose else lambda *a, **k: None

    for epoch in range(num_epochs):
        vprint("\nEpoch {}/{}".format(epoch + 1, num_epochs))
        vprint("-" * 12)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for samples, labels in tqdm(data_loader[phase], ncols=80, disable=not verbose):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    samples, labels = samples.to(device), labels[:, :2].to(device)
                    optimizer.zero_grad()
                    out = model(samples)
                    loss = calc_loss(out, labels, phase)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
            epoch_loss = running_loss / len(data_loader[phase].dataset)

            if phase == "test":

                t_preds, t_labels = [], []
                with torch.no_grad():
                    for samples, labels in data_loader[phase]:
                        samples, labels = samples.to(device), labels[:, :2].to(device)
                        out = model(samples)
                        out, labels = 1 * out > 0.5, label_encoder(labels.cpu()).to(device).squeeze()
                        t_preds.append(out.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                t_preds = np.concatenate(t_preds)
                t_labels = np.concatenate(t_labels).squeeze()

                acc = np.mean(sum(t_preds == t_labels)) * 100.0 / len(t_labels)
                vprint("Test Loss: {:.5f}        acc: {:.4f}".format(epoch_loss, acc))

                if epoch_loss < best:
                    best, rem = epoch_loss, patience
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    rem -= 1
                epoch_loss_history.append(epoch_loss)

            else:
                vprint("Train Loss: {:.5f}".format(epoch_loss))
        if not rem:
            break

    time_elapsed = time.time() - begin
    print("\nTraining complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_history


def cotrain_model(model, data_loader, optimizer, patience, num_epochs=100, verbose=False):

    begin = time.time()
    epoch_loss_history, rem = [], patience
    best_model_wts, best_acc = copy.deepcopy(model.state_dict()), 100.0
    vprint = print if verbose else lambda *a, **k: None

    for epoch in range(num_epochs):
        vprint("\nEpoch {}/{}".format(epoch + 1, num_epochs))
        vprint("-" * 12)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for eeg, mus in tqdm(data_loader[phase], ncols=80, disable=not verbose):
                eeg, elabel = eeg[0].to(device), eeg[1].to(device)
                mus, mlabel = mus[0].to(device), mus[1].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    optimizer.zero_grad()
                    feat1, feat2, pred1, pred2, dd_out, labels = model(eeg, mus)
                    loss = calc_multiloss(pred1, pred2, dd_out, elabel, mlabel, labels, phase)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
            epoch_loss = running_loss / len(data_loader[phase].dataset)

            if phase == "test":
                feat1, feat2, pred1, pred2, dd_out, label1, label2, labeld = get_cross_preds(
                    model, data_loader[phase]
                )
                label1, label2 = label_encoder(label1).numpy(), label_encoder(label2).numpy()
                retr = retrieve(feat1, feat2, label1, label2, k=len(feat1), method="emotion")
                vprint("Test  Loss: {:.4f} \tEEG->MUS: {:.2f} %".format(epoch_loss, 100 * np.mean(retr)))

                if epoch_loss < best_acc:
                    best_acc, rem = epoch_loss, patience
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    rem -= 1
                epoch_loss_history.append(epoch_loss)

            else:
                vprint("Train Loss: {:.4f}".format(epoch_loss))

        if not rem:
            break

    time_elapsed = time.time() - begin
    print("\nTraining complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_history
