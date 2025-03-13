from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Resample
import torch
import torch.nn.functional as F
import librosa
from librosa.util import normalize
import glob
# from python_speech_features.base import mfcc, logfbank
# from python_speech_features import delta
from os.path import join
import numpy as np
import math
import random
import scipy.signal as sig
from typing import List, Any
from collections import defaultdict
import time

def extract_windows(data, config):
    """
    :param data: waveform for one file  [Channel,T]
    :param config:
    :return:  overlapping windows stacked in batch_size axis   [N,C,WINDOW_SIZE]
    """
    if data.ndim == 1:
        data = data.unsqueeze(0)
    assert data.ndim == 2
    numWindows = int(math.ceil(float(data.size(-1)) / config.STEP_SIZE))
    windows = []
    offset = 0
    pad_len = 0
    for i in range(0, numWindows):
        frame = data[:, offset:offset + config.SEG_LENGTH]

        if frame.size(-1) < config.SEG_LENGTH:
            pad_len = config.SEG_LENGTH - frame.size(-1)
            frame = F.pad(frame, (0, pad_len),
                          'constant', value=0.0)
        windows.append(frame)
        offset += config.STEP_SIZE

    for i in range(len(windows)):
        windows[i] *= config.WINDOWING_MULT

    windows = torch.stack(windows).float()  # [N,C,WINDOW_SIZE]
    return windows, pad_len


def unpreprocess_waveform(waveform, maxabs):
    return np.copy(waveform) * maxabs

class AudioDataset4(torch.utils.data.Dataset):
    """
    return unnormalized waveform (without abs) and not random augment
    """
    def __init__(self, filelist, sampling_rate, resampling_rate=None, train=True, shuffle=False, max_duration_sec=4):

        self.audio_files = filelist

        if shuffle:
            random.seed(1234)
            random.shuffle(self.audio_files)
        self.train = train
        if resampling_rate is not None:
            self.sampling_rate = resampling_rate
            # self.resample = Resample(orig_freq=sampling_rate, new_freq=resampling_rate)
        else:
            self.sampling_rate = sampling_rate
            # self.resample = Resample(orig_freq=sampling_rate, new_freq=sampling_rate)

        self.max_length = self.sampling_rate * max_duration_sec if max_duration_sec is not None else None  #4 sec

        print('audio max length', self.max_length)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # start_time = time.time()
        waveform, sampling_rate = torchaudio.load(self.audio_files[index])
        if sampling_rate != self.sampling_rate:  
            waveform = torchaudio.functional.resample(waveform, sampling_rate, self.sampling_rate)

        length = waveform.size(-1)
        if self.max_length is not None:
            if length > self.max_length:
                start = random.randint(0, length - self.max_length - 1)
                waveform = waveform[:, start:start + self.max_length]
            elif length < self.max_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.max_length - length), "constant", 0)
        # duration = time.time() - start_time
        # print("loading duration:", duration)
        return waveform, length, 1.0, 0.0 



NUM_DATASET_WORKERS = 2
import csv
def read_metadata_jamendo(metadata_path, dataset_root):
    audio_path = []
    audio_dur_sec = []
    with open(metadata_path) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        next(tsv_file)
        for i, line in enumerate(tsv_file):
            audio_path.append(join(dataset_root, line[3]))
            audio_dur_sec.append(line[4])
    return audio_path, audio_dur_sec

def get_loader(config, datasetClass=None, ngpus=False, n_workers=NUM_DATASET_WORKERS):
    """
    get dataset according to config.dataset
    :param config:
    :return: train_loader test_loader
    """
    if config.dataset == 'timit':
        print('Use TIMIT DataSet')
        dataset_root = '/media/Dataset/timit/'
        train_path = join(dataset_root, 'data', 'TRAIN')
        test_path = join(dataset_root, 'data', 'TEST')
        trainset = glob.glob(join(train_path, '**/*.wav'), recursive=True)
        testset = glob.glob(join(test_path, '**/*.wav'), recursive=True)

    elif config.dataset == 'LibriSpeech':
        print('Use LibriSpeech DataSet')
        dataset_root = '/media/Dataset/LibriSpeech/'
        train_path = [join(dataset_root, 'train-clean-100'),
                      join(dataset_root, 'train-clean-360'),
                      # join(dataset_root, 'train-other-500'),
                     ]
        dev_path = join(dataset_root, 'dev-clean')
        test_path = join(dataset_root, 'test-clean')
        trainset = []
        for p in train_path:
            trainset += glob.glob(join(p, '**/*.flac'), recursive=True)
        testset = glob.glob(join(test_path, '**/*.flac'), recursive=True)
        devset = glob.glob(join(dev_path, '**/*.flac'), recursive=True)

    elif config.dataset == 'LibriTTS':
        print('Use LibriTTS DataSet')
        dataset_root = '/media/Dataset/LibriTTS/'
        dataset_root_test = '/media/Dataset/LibriTTS/'
        train_path = join(dataset_root, 'train-clean-100')
        test_path = join(dataset_root_test, 'test-clean')
        trainset = glob.glob(join(train_path, '**/*.wav'), recursive=True)
        testset = glob.glob(join(test_path, '**/*.wav'), recursive=True)

    elif config.dataset == 'Jamendo':
        print('Use Jamendo Music DataSet')
        dataset_root = "/media/Dataset/mtg-jamendo-dataset/audio"
        train_metadata = "/media/Dataset/mtg-jamendo-dataset/scripts/split-0/raw_30s_cleantags_50artists-train.tsv"
        dev_metadata = "/media/Dataset/mtg-jamendo-dataset/scripts/split-0/raw_30s_cleantags_50artists-validation.tsv"
        test_metadata = "/media/Dataset/mtg-jamendo-dataset/scripts/split-0/raw_30s_cleantags_50artists-test.tsv"
        trainset, _ = read_metadata_jamendo(train_metadata, dataset_root)
        devset, _ = read_metadata_jamendo(dev_metadata, dataset_root)
        testset, _ = read_metadata_jamendo(test_metadata, dataset_root)
        with open("/media/Dataset/mtg-jamendo-dataset/scripts/split-0/train_mono_file.txt") as file:
            while True:
                item = file.readline().rstrip('\n')
                if not item:
                    break
                trainset.remove(join(dataset_root, item))
        test_bad_item = ["95/426795.mp3", "67/428567.mp3", "15/575215.mp3", "33/249733.mp3","84/340484.mp3"]
        [testset.remove(join(dataset_root,item)) for item in test_bad_item]

    else:
        raise NotImplementedError('No dataset loader defined')

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    
    trainDataset = datasetClass(trainset, config.SAMPLE_RATE, config.Resample_rate,
                                shuffle=True, train=True)
    if ngpus:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset,
                                                    pin_memory=True,
                                                    batch_size=config.BATCH_SIZE,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset,
                                                    num_workers=n_workers,
                                                    pin_memory=True,
                                                    batch_size=config.BATCH_SIZE,
                                                    worker_init_fn=worker_init_fn_seed,
                                                    )
    devDataset = datasetClass(devset, config.SAMPLE_RATE, config.Resample_rate,
                                train=False, shuffle=False)
    testDataset = datasetClass(testset, config.SAMPLE_RATE, config.Resample_rate,
                                train=False, shuffle=False,
                                max_duration_sec=None)

    print('#LenTrainDataset', len(trainDataset))
    print('#LenDevDataset', len(devDataset))
    print('#LenTestDataset', len(testDataset))

    print('#LenTrainLoader', len(train_loader))

    if ngpus:
        return train_loader, train_sampler, devDataset, testDataset
    else:
        return train_loader, devDataset, testDataset


class default_Config:
    SAMPLE_RATE = 16000
    SEG_LENGTH = 1
    BATCH_SIZE = 8
