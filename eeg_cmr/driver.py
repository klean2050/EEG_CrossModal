import numpy as np, os, torch, warnings, argparse

warnings.filterwarnings("ignore")

from utils import *
from preprocess import *
from loaders import get_loader
from models import *
from train import pretrain_model, cotrain_model
from test import test_on, test_participant

parser = argparse.ArgumentParser(
    prog="EEG_Music_CrossModal_Retrieval", description="Configure EEG Experiment."
)
parser.add_argument("metric", type=str, help="Choose retrieval metric between P@10 / mAP")
parser.add_argument("mode", type=str, help="Choose experiment mode between train / test")
parser.add_argument("--pretrain", action="store_true", help="enable pre-training (default: False)")
parser.add_argument("--test_pretrain", action="store_true", help="test pre-training (default: False)")
parser.add_argument(
    "--aggregate", action="store_true", help="output results aggregated on track basis (default: False)"
)
parser.add_argument("--verbose", action="store_true", help="print training procedure (default: False)")
parser.add_argument(
    "--extract_feats", action="store_true", help="extract EEG input features (default: False)"
)
parser.add_argument(
    "--extract_embeds", action="store_true", help="extract MusiCNN embeddings (default: False)"
)

args = vars(parser.parse_args())

subjects = range(32)
patience, lr = 15, 1e-4

metric = args["metric"]
mode = args["mode"]
pretrain = args["pretrain"]
test_pretrain = args["test_pretrain"]
aggregate = args["aggregate"]
verbose = args["verbose"]
extract_feats = args["extract_feats"]
extract_embed = args["extract_embeds"]

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
        net_name = "EEG_net_{}.pt".format(dim)

        if pretrain:
            # pretrain using leave-one-subject-out cross validation
            print("\nPre-Training to test Participant {}.".format(i))
            ind_loader = get_loader(datapath, mode="eeg", batch_size=128, dur=3, subject=i)
            loader = {"train": ind_loader["train"], "test": ind_loader["valid"]}
            emodel = EEG_LSTM(num_feats=12, hidden=[128, 256]).to(device)
            optimizer = torch.optim.Adam(list(emodel.parameters()), lr=1e-4)
            emodel, _ = pretrain_model(emodel, loader, optimizer, patience, verbose=verbose)
            torch.save(emodel, p_dir + net_name)

        if test_pretrain:
            if "ind_loader" not in locals():
                ind_loader = get_loader(datapath, mode="eeg", batch_size=128, dur=3, subject=i)
            emodel = torch.load(p_dir + net_name)
            test_on(ind_loader["test"], emodel, aggregate)

        # cotrain using 5-fold cross validation
        print("\nModel of Participant {}.".format(i))
        crs_loader = get_loader(datapath, mode="cross", batch_size=32, dur=3, subject=i)

        num_folds = 5 if mode == "train" else 0
        for fold in range(num_folds):
            cnet_name = "CROSS_{}_{}.pt".format(fold, dim)
            loader = {"train": crs_loader["train{}".format(fold)], "test": crs_loader["test{}".format(fold)]}
            model = CROSS_NN(p_dir + net_name, dim=32, pretrained=False).to(device)
            optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
            model, _ = cotrain_model(model, loader, optimizer, patience, verbose=verbose)
            torch.save(model, p_dir + cnet_name)

        results = test_participant(p_dir, crs_loader, metric, aggregate)
        all_results += [score / len(subjects) for score in results]

    print("\n------------------------------------------------------------")
    print("Overall Stimulus Retrieval from EEG Queries ({}): {:.2f} %".format(metric, all_results[0]))
    print("Overall Related Track Retrieval from EEG Queries ({}): {:.2f} %".format(metric, all_results[1]))
    print("Overall EEG Emotion Classification (acc): {:.2f} %".format(all_results[2]))
    print("Overall MUS Emotion Classification (acc): {:.2f} %".format(all_results[3]))
