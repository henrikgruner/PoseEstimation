
import torch
import numpy as np
import os


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, sample_num=1024):
        super(ModelNetDataset, self).__init__()
        self.paths = [os.path.join(data_folder, i)
                      for i in os.listdir(data_folder)]
        self.sample_num = sample_num
        self.size = len(self.paths)
        print(f"dataset size: {self.size}")

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        pc = np.loadtxt(fpath)
        pc = np.random.permutation(pc)
        return pc[:self.sample_num, :].astype(float)

    def __len__(self):
        return self.size


data_folder = 'dataset/modelnet40_manually_aligned/airplane'
train_folder = os.path.join(data_folder, 'train_pc')
val_folder = os.path.join(data_folder, 'test_fix')
train_dataset = ModelNetDataset(train_folder, sample_num=1024)
batch_size = 1

dl_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

dl_eval = torch.utils.data.DataLoader(
    val_folder,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


pc = next(iter(dl_train))

print(pc.shape[0])
print(pc)


# BEST RUN004 -> 245 (0.4 mean geodesic error!) -> sofa?
# BEST RUN for chair?
