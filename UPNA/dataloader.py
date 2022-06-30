
import UPNA
import torch
import os
import numpy as np
'''
import torch
from dataloader import get_upna_loaders
import matplotlib.pyplot as plt
'''


def get_upna_loaders(batch_size, train_all, dataset_dir):
    dataset = UPNA.UPNA(dataset_dir)
    train_ds = dataset.get_train()
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval
