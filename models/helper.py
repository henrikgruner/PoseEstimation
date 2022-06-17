from resnet import resnet18, resnet50, resnet101, ResnetHead
from cnns import BaseNet
import torch.nn as nn
import torch
import sys
import os
from datetime import datetime
import glob


# for testing
sys.path.append('../SO3/saved_models_multiple')

dispatch = {'BaseNet': BaseNet, 'resnet18': resnet18,
            'resnet50': resnet50, 'resnet101': resnet101, 3: "3D", 6: "6D", 9: "SVD"}


def save_network(epoch, model, opt, model_name, path):
    opt_name = opt.__class__.__name__

    NAME = str(model_name) + '_state_dict_{}.pkl'.format(epoch)
    PATH = os.path.join(path, NAME)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, PATH)


def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch


def get_network(name, out_dim, num_classes=1):
    if('resnet' in name):
        base = dispatch[name](pretrained=True)
        network = ResnetHead(base, num_classes, 32, 512, out_dim)
    else:
        network = dispatch[name](3, out_dim)

    return network


if __name__ == '__main__':
    print(get_network('resnet101', 9, 1))
