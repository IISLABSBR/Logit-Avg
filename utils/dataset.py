import itertools
import os

import numpy as np
import pandas as pd
import random
import pickle


def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype = np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype = np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header = None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())

    return train_sessions, test_sessions, num_items


class Dataset:
    def __init__(self, sessions, sort_by_length=True):
        self.sessions = sessions
        index = create_index(sessions)
        if sort_by_length:
            # sort by label Index in descending order (label means length)
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        last_item = self.sessions[sid][lidx-1]
        label = self.sessions[sid][lidx]

        return seq, last_item, label

    def __len__(self):
        return len(self.index)




