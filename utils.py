import numpy as np, math, torch
from scipy.signal import butter, filtfilt

device, datapath = 'cuda:1', '/gpu-data/kavra/DEAP/'

def label_encoder(labels, ignore_segs=False):

    numl = labels.shape[-1]
    if labels.ndim == 1: labels = np.expand_dims(labels, axis=0)

    encoded = np.zeros((len(labels),50,numl-1)) if ignore_segs else np.zeros((len(labels),numl-1))
    for i, sample in enumerate(labels):
        if ignore_segs: sample = sample[0]
        if sample[0]<=5 and sample[1]<=5: encoded[i] = np.concatenate(([0],sample[2:numl]))
        if sample[0]<=5 and sample[1]> 5: encoded[i] = np.concatenate(([1],sample[2:numl]))
        if sample[0]> 5 and sample[1]<=5: encoded[i] = np.concatenate(([0],sample[2:numl]))
        if sample[0]> 5 and sample[1]> 5: encoded[i] = np.concatenate(([1],sample[2:numl]))

    return torch.Tensor(encoded).long()

def make_pairs(eeg, elabels, mus_samples, mus_labels, dur):

	num, segs = len(eeg), 60 - dur + 1
	mus, mlabels = np.zeros((num,128)), np.zeros((num,2))
	for i in range(num):
		track, seg = int(elabels[i,2]), int(elabels[i,4])
		mus[i], mlabels[i] = mus_samples[track*segs + seg], mus_labels[track*segs + seg]
	return mus, mlabels

def get_band(signal, band, fs=128):

        bands = {"theta": [2,4],
                 "alpha": [4,7],
                  "beta": [7,15],
                 "gamma": [15,23]}

        if band=='raw': return signal
        b,a = butter(5,bands[band],btype="bandpass",fs=fs)
        return filtfilt(b,a,signal)

def populate(signal, duration, overlap):
        '''
        signal   = 1D array you want to segment
        duration = number of samples based on sampling rate
        overlap  = number of overlapped samples at shift
        '''
        num_seg = int(math.ceil((len(signal)-overlap)/(duration-overlap)))
        tempBuf = [signal[i:i+duration] for i in range(0,len(signal),(duration-int(overlap)))]
        tempBuf[num_seg-1] = np.pad(tempBuf[num_seg-1],(0,duration-tempBuf[num_seg-1].shape[0]),'constant')
        return np.vstack(tempBuf[0:num_seg])
