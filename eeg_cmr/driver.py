import numpy as np, os, torch

from utils import *
from preprocess import *
from loaders import get_loader
from models import *
from train import pretrain_model, cotrain_model
from test import test_on, test_participant

subjects = range(32)
patience, lr = 15, 1e-4
metric = "mAP"
mode = "test"
pretrain = False
test_pretrain = False
aggregate = True
verbose = False
extract_feats = False
extract_embed = False

all_results = np.zeros(4)
if __name__ == "__main__":

    print("Program started. Utilized device: {}".format(device))

    if extract_feats:
        process_DEAP_stimuli(num_tracks=34, eeg_dur=3)
    if extract_embed:
        for i in subjects:
            process_DEAP_DE(datapath, str(i + 1), dur=3, exclude=True)

    for i in subjects:

        p_dir = datapath + "nets/nets_p{}/".format(i)
        os.makedirs(p_dir, exist_ok=True)

        if pretrain:
            # pretrain using leave-one-subject-out cross validation
            print("\nPre-Training to test Participant {}.".format(i))
            ind_loader = get_loader(datapath, mode="eeg", batch_size=128, dur=3, subject=i)
            loader = {"train": ind_loader["train"], "test": ind_loader["valid"]}
            emodel = EEG_LSTM(num_feats=12, hidden=[128, 256]).to(device)
            optimizer = torch.optim.Adam(list(emodel.parameters()), lr=1e-4)
            emodel, _ = pretrain_model(emodel, loader, optimizer, patience, verbose)
            torch.save(emodel, p_dir + "EEG_net.pt")

        if test_pretrain:
            if "ind_loader" not in locals():
                ind_loader = get_loader(datapath, mode="eeg", batch_size=128, dur=3, subject=i)
            emodel = torch.load(p_dir + "EEG_net.pt")
            test_on(ind_loader["test"], emodel, aggregate)

        # cotrain using 5-fold cross validation
        print("\nTraining on Participant {}.".format(i))
        crs_loader = get_loader(datapath, mode="cross", batch_size=32, dur=3, subject=i)

        num_folds = 5 if mode == "train" else 0
        for fold in range(num_folds):
            loader = {"train": crs_loader["train{}".format(fold)], "test": crs_loader["test{}".format(fold)]}
            model = CROSS_NN(p_dir, dim=32, pretrained=False).to(device)
            optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
            model, _ = cotrain_model(model, loader, optimizer, patience, verbose)
            torch.save(model, p_dir + "CROSS_{}_v.pt".format(fold))

        results = test_participant(p_dir, crs_loader, metric, aggregate)
        all_results += [score / len(subjects) for score in results]

    print("\n------------------------------------------------------------")
    print("Overall Stimulus Retrieval from EEG Queries ({}): {:.2f} %".format(metric, all_results[0]))
    print("Overall Related Track Retrieval from EEG Queries ({}): {:.2f} %".format(metric, all_results[1]))
    print("Overall EEG Emotion Classification (acc): {:.2f} %".format(all_results[2]))
    print("Overall MUS Emotion Classification (acc): {:.2f} %".format(all_results[3]))
