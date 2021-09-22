import numpy as np, math, pickle, os, time, copy
from scipy.signal import stft, butter, filtfilt
#from musicnn.extractor import extractor
from tqdm import tqdm

from utils import *


def load_data(path, only_eeg=True, exclude=False, participant="1"):

        # access participant's trials
        if len(participant)==1: participant = "0"+participant
        f = pickle.load(open(path+"data_preprocessed/s"+participant+".dat",'rb'), encoding='latin1')

        # exclude trials of the eliminated stimuli
        songs = list(range(40))
        for s in range(40):
            if not exclude: break
            if s in [7,9,10,15,33,35]: songs.remove(s)

        # extract the desired data
        data = f["data"][songs,:32,3*128:] if only_eeg else f["data"]
        labels = f["labels"][songs,:2]

        # assign each trial its track and participant index
        labels = np.c_[labels, list(range(len(songs))), (int(participant)-1)*np.ones(len(songs))]
        return np.array(data), np.array(labels)
        
def process_DEAP_DE(path, p, dur=1):

        p = '0'+str(p) if int(p)<10 else str(p)
        fs = 128; print('Request to process Participant {}.'.format(p),end='\r')
        data, labels = load_data(path, participant=p)

        # input feature extraction
        segs = 60 - dur + 1
        feat_vector = np.zeros((40,segs,32,4,dur))
        for song in range(data.shape[0]):        
                for channel in range(data.shape[1]):
                        for b, band in enumerate(['theta', 'alpha', 'beta', 'gamma']):
                                
                                signal = get_band(data[song,channel], band=band)
                                f, t, magn = stft(signal, fs=fs, window='hann', nperseg=fs, noverlap=0) # (65,61)
                                feats = [-0.5*np.log((abs(val)**2).mean()) for val in magn.T[1:]]

                                segments = populate(np.arange(60), duration=dur, overlap=dur-1) # (58,3)
                                for n, seg in enumerate(segments): feat_vector[song, n, channel, b] = np.array(feats)[seg]

        folder = path+'{}sec_de'.format(dur)
        if not os.path.exists(folder): os.makedirs(folder)
        
        final_data = feat_vector.reshape(40*segs, 32, 4*dur)
        final_labels = np.repeat(labels, segs, axis=0)
        final_labels = np.c_[final_labels, 40*(list(range(segs)))]

        np.save('{}/P{}_feats.npy'.format(folder,p),final_data)
        np.save('{}/P{}_annot.npy'.format(folder,p),final_labels)
        print('Successfully processed Participant {}.'.format(p))

def process_DEAP_stimuli(num_tracks, eeg_dur):

        segs, fs = 60 - eeg_dur + 1, 44100
        path = datapath + 'stimuli/'
        start = np.zeros(num_tracks,)
        labels = np.zeros((num_tracks, 2))
        
        # load labels and starting second of trials
        with open(path+"video_list_labels.csv") as f:
                csv_reader = csv.DictReader(f, delimiter=',')
                for i, row in enumerate(csv_reader):
                        labels[i] = [row["AVG_Valence"], row["AVG_Arousal"]]
                        start[i] = int(row["Start"])

        print('\nLoading DEAP stimuli tracks...')
        final_data = np.zeros((num_tracks, segs, 128))
        for i, track in enumerate(sorted(os.listdir(path))):
                if 'wav' not in track: continue
                features = extractor(path+track, model='MSD_vgg', input_overlap=2, extract_features=True)[2]
                features = features['pool5']
                # isolate the minute of interest
                final_data[i] = features[start[i]:start[i]+60]
                print("{}/34 tracks processed.".format(i+1), end='\r')
                
                
        final_data = final_data.reshape(num_tracks*segs, -1)
        np.save(path + "tracks_embeds.npy", final_data)
        labels = np.repeat(labels, segs, axis=0)
        np.save(path + "tracks_labels.npy", labels)
