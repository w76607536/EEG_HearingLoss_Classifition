# -*- coding: utf-8 -*-

'''
@Time    : 12/1/2022 1:46 下午
@Author  : bobo
@FileName: EEGDataProcess.py
@Software: PyCharm

'''
import os
import numpy as np
import mne
from matplotlib import pyplot as plt
from autoreject import AutoReject
import sys
import argparse
from mne.baseline import rescale
from mne.viz import centers_to_edges
#from glob import glob



# %matplotlib qt
def parse_args():
    parser = argparse.ArgumentParser()

    #param = {'sfreq': 1000, 'l_freq': 0.1, 'h_freq': 30.0, 'tmin': -0.099, 'tmax': 0.5,'rejection_eeg': None, 'autoReject': True}
    

    parser.add_argument('--sfreq', default=None,
                        type=int,
                        help='sfreq: int(default: None/1000)')

    parser.add_argument('--l_freq', default=0.1,
                        type=float,
                        help='l_freq: (default: None/0.1)')

    parser.add_argument('--h_freq', default=30,
                        type=float,
                        help='l_freq: (default: None/30)')

    parser.add_argument('--tmin', default=-0.1,
                        type=float,
                        help='tmin: (default: -0.1)')


    parser.add_argument('--tmax', default= 0.5,
                        type=float,
                        help='tmax: (default: 0.499)')

    parser.add_argument('--rejection_eeg', default=0.0002,
                        type=float,
                        help='rejection_eeg: (default: None)')

    parser.add_argument('--autoReject',default=False,
                        type=bool,
                        help='autoReject: False')

    parser.add_argument('--picks',default=[5, 10, 23, 28, 35, 103, 110, 123, 129],
                        type=list,
                        help='autoReject: False')


    args = parser.parse_args()

    return args

class DataProcess(object):
    def __init__(self, filePath = None, args = None):
        #print(param)
        if filePath:
            self.filePath = filePath
            self.fileType = self.filePath.split('.')[-1]
        else:
            self.filePath = None
            self.fileType = None

        self.sfreq = args.sfreq
        self.l_freq = args.l_freq
        self.h_freq = args.h_freq
        self.tmin = args.tmin
        self.tmax = args.tmax
        if self.tmin and self.tmax:
            self.xmin = int(1000 * self.tmin)
            self.xmax = int(1000 * self.tmax)
        else:
            self.xmin = None
            self.xmax = None
        self.rejection_eeg = args.rejection_eeg
        self.autoReject = args.autoReject
        self.picks = args.picks

        self.raw = None

        self.epochs = None

        self.data_mmn = None

        self.mean_dataS = None

        self.mean_dataD = None

    def setFilePath(self, filePath):

        self.filePath = filePath
        self.fileType = self.filePath.split('.')[-1]


    def preprocessing(self):
		#降采样
        self.raw.load_data()
        if self.sfreq:
            self.raw.resample(sfreq=self.sfreq)

		# 滤波
        self.raw.filter(l_freq=self.l_freq,h_freq=self.h_freq)

        self.raw.set_eeg_reference(ref_channels='average')


		# epoch & reject
        event_id = {'stad': 6, 'devt': 7}
        events = mne.find_events(self.raw, stim_channel='STI 014')
        self.epochs = mne.Epochs(self.raw, events,event_id=event_id, reject= None,tmin=self.tmin, tmax=self.tmax)
        #mne.time_frequency.tfr_array_morlet(self.epochs)

        return True


    def rejection(self):
		#autoReject
        self.epochs.load_data()
        if self.autoReject:
            ar = AutoReject()
            self.epochs = ar.fit_transform(self.epochs)
        elif self.rejection_eeg:
			#reject
			#print (self.rejection_eeg)
            reject_criteria = { 'eeg':float(self.rejection_eeg) }  # 150 µV)
			#print(reject_criteria)
            self.epochs.drop_bad(reject_criteria)

        return True

    def save_epochs(self,event):

        self.epochs = self.epochs.apply_baseline((-0.1, 0))
        epochs_data = self.epochs[event].get_data(tmin = self.tmin, tmax= self.tmax)
        #stad_epochs_data = self.epochs['stad'].get_data(picks=[128],tmin=self.tmin, tmax=self.tmax)
        #self.data = np.mean(stad_epochs_data, axis=0).ravel() * 1e6
        #self.showImage(self.data)
        #freqs = mne.time_frequency.stftfreq(100,500)+100
        freqs = np.arange(3,30.,2)
        #data_0 = mne.time_frequency.stft(epochs_data[0], 400, tstep=2)

        #power = mne.time_frequency.tfr_array_morlet(epochs_data, sfreq=1000, freqs=freqs)
        power = mne.time_frequency.tfr_morlet(self.epochs[event], freqs=freqs, n_cycles = 1, return_itc = False,picks=[128])

        #plt.plot(data_0[0])
        #plt.show()
        
        #print(power.size)
        #power.plot([0], baseline=(0., 0.1), mode='mean', vmin=self.tmin, vmax=self.tmax, show=False, colorbar=False)
        power.plot([-1])
        #plt.show()

    def showImage(self, data):
        t = range(int(self.tmin*1000), int(self.tmax*1000), 1)
        max = np.max(data)
        min = np.min(data)
        if max < -min:
            max = -min

        axes = plt.gca()
        plt.plot(t, data, color='r', label='MMN')
        plt.vlines(0, -max, max, color='#000000', linestyles="dashed")
        plt.hlines(0, self.xmin, self.xmax, color='#000000', linestyles="dashed")
        plt.legend()
        axes.set_xlim(left=self.xmin, right=self.xmax)
        axes.set_ylim(top=max, bottom=-max)
        plt.xlabel('time(ms)')
        plt.ylabel('μv')
        plt.show()


    def getRaw(self):
        self.raw = mne.io.read_raw_egi(self.filePath)
        #print(self.raw.info)
        return self.raw.info


def main():
    args = parse_args()

    #filepaths = glob(r'../../Data/*')

    filePath = 'F:\\Git_project\\EEGProcessing\\Data\\xuzhirou_puretone_20180814_020047.mff'


    dataProcess = DataProcess(args = args)

    dataProcess.setFilePath(filePath)

    dataProcess.getRaw()

    dataProcess.preprocessing()

    dataProcess.rejection()

    dataProcess.save_epochs("stad")

    dataProcess.save_epochs("devt")



if __name__ == '__main__':
    
    main()

