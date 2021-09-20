import numpy as np, math, pickle, os, time, copy
from scipy.signal import stft, butter, filtfilt
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
        feat_vector = np.zeros((40,32,51,4*10))
        for song in range(data.shape[0]):        
                for channel in range(data.shape[1]):
                    segments = populate(data[song,channel], duration=dur*fs, overlap=(dur-1)*fs)
                    for seg in range(segments.shape[0]):
                        feats = []
                        for band in ['theta', 'alpha', 'beta', 'gamma']:
                            signal = get_band(segments[seg], band=band)
                            f, t, magn = stft(signal, fs=fs, window='hann', nperseg=fs, noverlap=0)
                            feats += [-0.5*np.log((abs(val)**2).mean()) for val in magn.T[1:]]
                        feat_vector[song, channel, seg] = feats

        folder = path+'{}sec_de'.format(dur)
        if not os.path.exists(folder): os.makedirs(folder)

        np.save('{}/P{}_feats.npy'.format(folder,p),feat_vector)
        np.save('{}/P{}_annot.npy'.format(folder,p),labels)
        print('Successfully processed Participant {}.'.format(p))

