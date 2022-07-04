import torch.distributed as dist
import sys
sys.path.append('../data')
from dataset import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
import time
import random
from model import ResnetRS

device = 'cuda'
def symmetric_orthogonalization(x):
    """
    Code from https://github.com/amakadia/svd_for_pose
    Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
    x: should have size [batch_size, 9]
    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r

def loss_frobenius(R_pred, R_true):
    difference = R_true - R_pred
    frob_norm = torch.linalg.matrix_norm(difference, ord='fro')
    return frob_norm.mean()


dataset = 'SO3'
dataset_dir = '../data/datasets/'
seed =random.seed(10)
model_name = 'resnetrs101'
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, batch_size=4, pin_memory=False, num_workers=0):

    classes = ['bathtub']
    dataset_train = ModelNet10SO3(dataset_dir, classes, True)


    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=train_sampler)

    return trainloader

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    trainloader = prepare(rank, world_size)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = ResnetRS.create_pretrained(
    model_name, in_ch=3, num_classes=1).to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    epochs = 200
    print('hore')
    for epoch in range(epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        trainloader.sampler.set_epoch(epoch)       

        model.train()
        for step, (img, ex, _, _, _, _) in enumerate(trainloader):
            optimizer.zero_grad(set_to_none=True)
            R_pred =symmetric_orthogonalization(model(img))
            R = ex[:, :3, :3].to(rank)
            loss = loss_frobenius(R, R_pred)
            loss.backward()
            optimizer.step()

            if step % 1000 == 0 and rank == 0:
                print('Epoch: {} step: {} testloss: {}'.format(epoch, step, loss.item()))     
                exit()   
    cleanup()





import torch.multiprocessing as mp
if __name__ == '__main__':
    time = time.time()
    # suppose we have 3 gpus
    world_size = 2
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
    print(time.time()-time())