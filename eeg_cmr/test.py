import numpy as np, torch, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

from utils import *


def test_on(test_loader, model, aggregate=False):

    t_preds, t_labels = [], []
    with torch.no_grad():
        for samples, labels in test_loader:
            samples, labels = samples.to(device), labels[:, :2].to(device)
            out = model(samples)
            out, labels = 1 * out > 0.5, label_encoder(labels.cpu()).to(device).squeeze()
            t_preds.append(out.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
    t_preds = np.concatenate(t_preds)
    t_labels = np.concatenate(t_labels).squeeze()

    acc = (
        (np.mean(sum(t_preds == t_labels)) * 100.0 / len(t_labels))
        if not aggregate
        else track_aggregate(t_preds, t_labels, task="class", tracks=34)
    )
    print("Test Accuracy: {:.2f}%".format(acc))


def get_cross_preds(model, loader):

    t_eeg, t_mus, p_eeg, p_mus, d_out, e_lab, m_lab, d_lab = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for eeg, mus in loader:
            feat1, feat2, pred1, pred2, dd_out, labels = model(eeg[0].to(device), mus[0].to(device))
            t_eeg.append(feat1.cpu().numpy())
            t_mus.append(feat2.cpu().numpy())
            p_eeg.append(pred1.cpu().numpy())
            p_mus.append(pred2.cpu().numpy())
            d_out.append(dd_out.cpu().numpy())
            e_lab.append(eeg[1].cpu().numpy())
            m_lab.append(mus[1].cpu().numpy())
            d_lab.append(labels.cpu().numpy())

        t_eeg = np.vstack(t_eeg)
        t_mus = np.vstack(t_mus)
        p_eeg = np.hstack(p_eeg)
        p_mus = np.hstack(p_mus)
        d_out = np.hstack(d_out)
        e_lab = np.vstack(e_lab)
        m_lab = np.vstack(m_lab)
        d_lab = np.hstack(d_lab)

    return t_eeg, t_mus, p_eeg, p_mus, d_out, e_lab, m_lab, d_lab


def retrieve(eeg, mus, elabels, mlabels, k=0, method="track"):

    dist = cdist(eeg, mus, "cosine")
    sort, res = dist.argsort(), []
    for i in range(dist.shape[0]):
        order, r, p = sort[i], 0.0, 0.0
        for j in range(k):
            if (method == "emotion" and elabels[i, 0] == mlabels[order[j], 0]) or (
                method == "track" and elabels[i, 1] == mlabels[order[j], -1]
            ):
                r += 1.0
                p += r / (j + 1)
        res += [p / r] if r else [0]  # mAP
    return res


def track_aggregate(preds, labels, task="retr", tracks=7):

    segments, tracks, correct = 58, tracks, []
    sorting = labels[:, 1].argsort()
    preds = np.array(preds).take(sorting)
    labels = labels[:, 0].take(sorting)

    for i in range(tracks):
        trial_indices = np.arange(segments * i, segments * (i + 1))
        trial, label = preds[trial_indices], labels[trial_indices]
        result = 1 * (sum(trial == label) > segments / 2) if task == "class" else np.median(trial)
        correct.append(result)
    return 100 * np.mean(correct)


def visualize(features, labels, p, name):

    tsne = TSNE(
        n_components=2, perplexity=50, learning_rate=130, metric="cosine", square_distances=True
    ).fit_transform(features)
    tx = MinMaxScaler().fit_transform(tsne[:, 0].reshape(-1, 1))[:, 0]
    ty = MinMaxScaler().fit_transform(tsne[:, 1].reshape(-1, 1))[:, 0]

    fig = plt.figure()
    plt.rcParams["font.size"] = 10
    ax = fig.add_subplot(111)
    colors = ["red", "blue", "crimson", "steelblue"]
    for label in range(2):
        # find the samples of this class
        indices = [i for (i, l) in enumerate(labels) if l == label]
        ln = int(len(indices) / 2)
        # EEG points
        curr_tx, curr_ty = np.take(tx, indices[:ln]), np.take(ty, indices[:ln])
        ax.scatter(curr_tx, curr_ty, c=colors[label], marker=".", label=str(label) + " - EEG")
        # MUS points
        curr_tx, curr_ty = np.take(tx, indices[ln:]), np.take(ty, indices[ln:])
        ax.scatter(curr_tx, curr_ty, c=colors[label + 2], marker="*", label=str(label) + " - Music")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best")

    dist_dir = datapath + "tsne/tsne_p{}/".format(p)
    os.makedirs(dist_dir, exist_ok=True)
    fig.savefig(dist_dir + name)
    plt.close()


def test_fold(p_dir, trial, loader, metric, aggregate=False):

    model_name = p_dir + "CROSS_{}_{}.pt".format(trial, dim)
    model = torch.load(model_name, map_location=device)
    model.eval()

    feats1, feats2, preds1, preds2, _, labels1, labels2, _ = get_cross_preds(model, loader)
    labels1 = label_encoder(labels1).numpy()
    labels2 = label_encoder(labels2).numpy()
    preds1 = 1 * (preds1 > 0.5)
    preds2 = 1 * (preds2 > 0.5)

    # t-SNE of common latent space
    features = np.concatenate((feats1, feats2))
    labels = np.concatenate((labels1[:, 0], labels2[:, 0]))
    subject = p_dir.split("p")[-1].split("/")[0]
    visualize(features, labels, p=subject, name="CROSSD_{}_{}".format(trial, dim))

    k = 10 if metric == "P@10" else len(feats1)

    retr_track = retrieve(feats1, feats2, labels1, labels2, k, method="track")
    if not aggregate:
        retr_track = 100 * np.mean(retr_track)
    else:
        retr_track = track_aggregate(retr_track, labels1, task="retr")

    retr_emot1 = retrieve(feats1, feats2, labels1, labels2, k, method="emotion")
    if not aggregate:
        retr_emot = 100 * np.mean(retr_emot1)
    else:
        retr_emot = track_aggregate(retr_emot1, labels1, task="retr")

    if not aggregate:
        pred_eeg = 100 * sum(preds1 == labels1[:, 0]) / len(preds1)
    else:
        pred_eeg = track_aggregate(preds1, labels1, task="class")

    if not aggregate:
        pred_mus = 100 * sum(preds2 == labels2[:, 0]) / len(preds2)
    else:
        pred_mus = track_aggregate(preds2, labels2, task="class")

    return [retr_track, retr_emot, pred_eeg, pred_mus], labels1, retr_emot1


def test_participant(p_dir, loaders, metric, aggregate=False):

    results = np.zeros(4)
    for fold in range(5):
        fold_results, labels1, retr_emot1 = test_fold(p_dir, fold, loaders["test{}".format(fold)], metric, aggregate)
        results += [score / 5 for score in fold_results]

    print("\nStimulus Retrieval from EEG Queries ({}): {:.2f} %".format(metric, results[0]))
    print("Related Track Retrieval from EEG Queries ({}): {:.2f} %".format(metric, results[1]))
    print("EEG Emotion Classification (acc): {:.2f} %".format(results[2]))
    print("MUS Emotion Classification (acc): {:.2f} %".format(results[3]))

    return results
