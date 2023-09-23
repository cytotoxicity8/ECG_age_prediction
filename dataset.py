import os
from glob import glob

import torch
import torch.utils.data

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from tqdm import tqdm



def butterworth_transform(ecg, order=4):
    fs = 500 #frequency
    lowcut = 0.5
    highcut = 50.0

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    #b, a = butter(order, low, btype='high')

    ecg= np.array([filtfilt(b, a, lead) for lead in ecg])

    return ecg



class SNUH_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path="data", ecg_filename = None, info_filename = "submission.csv", butterworth = False):

        self.info = pd.read_csv(os.path.join(data_path, info_filename))
        self.ecg = np.load(os.path.join(data_path, ecg_filename))
        self.index_list = list(self.ecg.keys())
        self.info.set_index('FILENAME', inplace=True)

        if butterworth:
            print("Butterworth filter")
            filtered_ecg = {}

            for ecg_key in tqdm(self.ecg.keys()):
                filtered_ecg[ecg_key] = butterworth_transform(self.ecg[ecg_key].reshape(12, 5000))

            self.ecg = filtered_ecg


    def __getitem__(self, index):

        idx = self.index_list[index]
        ecg = self.ecg[idx].reshape(12, 5000)

        return ecg, idx

    
    def __len__(self):
        return len(self.index_list)
  