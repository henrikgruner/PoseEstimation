import yaml
import glob
import os
import torch
'''
###'############
#mapping
classes = ['bathtub', 'bed', 'chair', 'desk','dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

#other way
mapping = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9}

##########
'''



rot_dis = {"SVD": ("SVD", 9), "6D": (
    "6D", 6), "5D": ("5D", 5), "quat": ("quat",4)}


config = "configs/resnet.yaml"


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def config_parser(args, verbose=True):
    config = read_yaml(args.dir + args.c)

    if(verbose):
        print("\n")
        print("PARAMETERS:")
        for arg in vars(args):
            print("     ", arg.ljust(12), getattr(args, arg))

        for i in config.keys():
            print('\n', i)
            for j in config[i].keys():
                print("     ", j.ljust(12), config[i][j])
        print('\n')

    # General section
    lr = config['GENERAL']['LR']
    batch_size = config['GENERAL']['BATCH_SZ']
    opt_name = config['GENERAL']['OPT']
    rot_rep, rot_dim = rot_dis[config["GENERAL"]["ROTREP"]]
    epochs = config['GENERAL']['EPOCHS']
    loss_fn = config['GENERAL']['LOSS']
    drop_epochs = config['GENERAL']['DROP_EPOCHS']
    save_interval = config['GENERAL']['SAVE_INTERVAL']
    # Network
    model_name = config["NETWORK"]["NAME"]

    # Dataset
    classes = config['DATASET']['CLASSES']
    dataset_dir = config['DATASET']['DIR']
    num_classes = len(classes)

    resume = config['RESUME_TRAINING']['RESUME']
    try:
        schedule = config['GENERAL']['LR_SCHEDULE']
    except Exception as e:
        schedule = False

    return lr, batch_size, opt_name, model_name, classes, rot_rep, rot_dim, epochs, drop_epochs, loss_fn, num_classes, dataset_dir, resume, save_interval, schedule


def load_network_parser(args):

    config = read_yaml(args.dir + args.c)
    config = config['RESUME_TRAINING']
    PATH = config['PATH']
    curr_epoch = config['CURR_EPOCH']
    return PATH, curr_epoch


def resume_train(model_name, args):
    PATH, curr_epoch = load_network_parser(args)
    NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
    LOAD_PATH = os.path.join(PATH, NAME)
    return LOAD_PATH, curr_epoch

def get_new_dir(rot_rep):
    DIR = rot_rep + '/logs'
    n = len(glob.glob(DIR + '/run*'))
    NEW_DIR = 'run' + str(n + 1).zfill(3)
    SAVE_PATH = os.path.join(DIR, NEW_DIR)

    # create new directory
    PATH = 'saved_models'
    MODEL_SAVE_PATH = os.path.join(SAVE_PATH, PATH)
    if not os.path.isdir(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    return SAVE_PATH, MODEL_SAVE_PATH

def cuda_confirm():

    device = torch.device("cuda" if(
    torch.cuda.is_available()) else "cpu")
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]

    print("cuda: ", torch.cuda.is_available())
    print("count: ", torch.cuda.device_count())
    print("names: ", device_names)
    print("device ids:", devices)
    return device, devices
