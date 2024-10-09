import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class Cifar10Dataset(Dataset):
    def __init__(self, fnames: List[str], labels: List[int] = None, transform=None):
        self.labels = labels
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = self.labels[idx] if self.labels is not None else None

        img = Image.open(fname)
        img = np.array(img, dtype=float)
        img = np.transpose(img, (2, 0, 1)) / 255
        return fname, torch.tensor(img), label


def collate_fn(batch_items):
    fnames = [x[0] for x in batch_items]
    images = [x[1] for x in batch_items]
    labels = [x[2] for x in batch_items]

    images = torch.stack(images, dim=0).type(torch.float)
    labels = torch.tensor(labels) if labels[0] is not None else labels
    return fnames, images, labels


def load_data(train_val_csv, train_val_img_dir, test_img_dir, batch_size, num_workers, num_samples):
    labels_map = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9,

    }

    train_val_df = pd.read_csv(train_val_csv)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2)
    train_ids, train_labels = train_df.id.values, train_df.label.apply(lambda x: labels_map[x]).astype(int).values
    val_ids, val_labels = val_df.id.astype(int).values, val_df.label.apply(lambda x: labels_map[x]).astype(int).values

    train_fnames = [os.path.join(train_val_img_dir, f"{img_id}.png") for img_id in train_ids]
    val_fnames = [os.path.join(train_val_img_dir, f"{img_id}.png") for img_id in val_ids]

    test_ids = [x.split('.')[0] for x in os.listdir(test_img_dir)]
    test_fnames = [os.path.join(test_img_dir, f"{img_id}.png") for img_id in test_ids]

    train_dataset = Cifar10Dataset(train_fnames[:num_samples], train_labels[:num_samples])
    val_dataset = Cifar10Dataset(val_fnames[:num_samples], val_labels[:num_samples])
    test_dataset = Cifar10Dataset(test_fnames[:num_samples])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader