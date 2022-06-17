
from get_initial_estimate import get_Rinit_from_net
import torch
from utility import *
from loss import *
from dataset import get_6D_loader, get_6D_eval_loader
from model.base import BasicBlock, Bottleneck
from functools import partial
from render_utility import *
from torch.utils.tensorboard import SummaryWriter

# MAIN

'''
Last inn bilder->
Last inn modeller->
Random eller nettverk for init?
# image, Extrinsic, Extrinsic_init, Class, Cad_id
# img_gt og img_init er tensor.
# torch.cat([img_gt, img_init])
Nettverk:
'''


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


def train_net(model, opt, dl_train, device, writer_train, iterations=1, lossfunc=None):
    '''
        input:
        model : network
        opt : optimizer
        criterion : loss function
        dl_train : dataloader with training data
    '''
    model.train()
    epoch_loss = []
    for index, (images, curr_images, depth, verts, ex, ex_curr, class_id, cad_id) in enumerate(dl_train):

        # Burde hente mesh points her -> Flere iterasjoner?
        #mesh_points = get_mesh_points(cad_id, object_class, ex_curr, batch_size)

        # Må HA begge bildene
        # Mesh for loss
        # Rendered
        opt.zero_grad()
        # curr_img, mesh = render_to_batch_parallel(
        #    cad_id, class_id, ex_curr, batch_size)

        # Må passe på at det kommer i batch
        # HER MÅ imgs tranformeres
        gt_img = images.to(device, dtype=torch.float)
        curr_images = curr_images.to(device, dtype=torch.float)

        verts = verts.to(device)
        ex = ex.to(device)
        ex_curr = ex_curr.to(device)
        depth = depth.to(device)

        model_input = torch.cat(
            [gt_img, depth.unsqueeze(dim=1), curr_images], dim=1)

        out = model(model_input)

        Rotation = out[:, :9]
        t = out[:, 9:12]

        # Parametrization to SE(3) with SVD
        R = symmetric_orthogonalization(Rotation)

        # er denne gudd?
        ex_curr = SE3_parameterization(R, t, ex_curr.to(device))

        # Få inn symmetri her a

        loss = compute_disentangled_ADD_L1_loss(
            ex_curr.to(device), ex.to(device), verts)

        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
        writer_train.add_scalar('Loss/iteration', loss.item(), index)

    return epoch_loss


# Brief setup
rot_rep = 'SVD'
rot_dim = 9

num_classes = 1
batch_size = 8

dataset_dir = 'data/'
dataset = ''
epochs = 300
drop_epochs = []
save_interval = 5
model_name = 'resnetrs101'
ngpu = 4
lr = 0.2


curr_epoch = 0

#lossfunc =GeodesicLoss()
lossfunc = 'ADDL1'
DIR = 'logs'
# automatically find the latest run folder
n = len(glob.glob(DIR + '/run*'))
NEW_DIR = 'run' + str(n + 1).zfill(3)

SAVE_PATH = os.path.join(DIR, NEW_DIR)
# create new directory

PATH = 'saved_models'
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, PATH)
if not os.path.isdir(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

device = torch.device("cuda" if(
    torch.cuda.is_available() and ngpu > 0) else "cpu")
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]

print("cuda: ", torch.cuda.is_available())
print("count: ", torch.cuda.device_count())
print("names: ", device_names)
print("device ids:", devices)

classes = ['toilet', 'sofa', 'chair', 'airplane', 'bathtub']

dl_train, dl_eval = get_6D_loader(
    dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False)

# model = ResnetRS.create_pretrained(
#    model_name, in_ch=6, num_classes=num_classes)
block = Bottleneck
layers = [3, 4, 6, 3]
model = ResnetRS.create_model(block, layers, num_classes=12, in_ch=7,
                              stem_width=64, down_kernel_size=1,
                              actn=partial(nn.ReLU, inplace=True),
                              norm_layer=nn.BatchNorm2d, seblock=True,
                              reduction_ratio=0.25, dropout_ratio=0.,
                              stochastic_depth_rate=0.0,
                              zero_init_last_bn=True)


model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=lr)


writer_train = SummaryWriter(
    log_dir=os.path.join(SAVE_PATH, 'train'), comment=f"_{model_name}_{opt.__class__.__name__}_{lr}_train")

for e in range(curr_epoch, epochs):
    # hva må gjøres hver epoke?
    verbose = e % int(save_interval) == 0 or e == (epochs - 1)

    epoch_time = time.time()

    if e in drop_epochs:
        lr *= 0.1
        for g in opt.param_groups:
            g['lr'] = lr
    train_loss = train_net(model, opt, dl_train, device, writer_train,
                           iterations=1, lossfunc=lossfunc)

    average_train_loss = (sum(train_loss) / len(train_loss))
    #average_eval_loss = (sum(val_loss) / len(val_loss))

    writer_train.add_scalar('Loss/train', average_train_loss, e)
    #writer_train.add_scalar('Loss/test', average_eval_loss, e)

    print(e, "/", epochs, " Training loss: ", average_train_loss,
          "Validation loss: ", 0, "time: ", epoch_time)

    if(verbose):
        save_network(e, model, opt, model_name, MODEL_SAVE_PATH)
