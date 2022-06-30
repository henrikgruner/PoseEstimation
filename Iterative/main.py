
from get_initial_estimate import get_Rinit_from_net
import torch
from utility import *
from loss import *

from dataset import get_6D_loader, get_6D_eval_loader,get_6D_loader_pregen
from model.base import BasicBlock, Bottleneck
from functools import partial
from render_utility import *
from torch.utils.tensorboard import SummaryWriter
from test import plot, combine_imgs
import multichannel_resnet
from multichannel_resnet import get_arch as Resnet

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

def angle_error(t_R1, t_R2):
    ret = torch.empty((t_R1.shape[0]), dtype=t_R1.dtype, device=t_R1.device)
    rotation_offset = torch.matmul(
        t_R1.transpose(1, 2).double(), t_R2.double())
    tr_R = torch.sum(rotation_offset.view(-1, 9)
                     [:, ::4], axis=1)  # batch trace
    cos_angle = (tr_R - 1) / 2
    if torch.any(cos_angle < -1.1) or torch.any(cos_angle > 1.1):
        raise ValueError(
            "angle out of range, input probably not proper rotation matrices")
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle * (180 / np.pi)
    
def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch


def sanity(gt_img, init_imgs, gt_ex, init_ex, model_ex, cad_ids, class_ids):

    for i, img in enumerate(gt_img):

        model_img, _ = render_from_id(
            cad_ids[i].item(), class_ids[i], model_ex[i], train=False)

        img = img.transpose(1, 2, 0)

        init_img = init_imgs[i]
        init_img = init_img.transpose(1, 2, 0)

        combined_img = combine_imgs(img, init_img)
        combined_img2 = combine_imgs(img, model_img)

        plot(img, init_img, model_img, str(i))
        plot(img, combined_img, combined_img2, str(i + 100))


def train_net_pregren(model, opt, dl_train, device, writer_train, iterations=1, lossfunc=None):

    model.train()
    epoch_loss = []
    angle_errors = []
    for index, (images, curr_images,render_img, depth, verts, ex, ex_curr, mod_ex, class_id, cad_id) in enumerate(dl_train):
        for it in range(iterations):
            opt.zero_grad()

            ex = ex.to(device)
            verts = verts.to(device)
            gt_img = images.to(device, dtype=torch.float)

            if(it == 0):
                ex_curr = ex_curr.to(device)

            if(it > 0):
                curr_images = render_img
                
                ex_curr = mod_ex.to(device)

            curr_images = curr_images.to(device, dtype=torch.float)


            depth = depth.to(device)
            # ALL TO GPU
            

            # Stack the inputs: image, depth, and initial guess
            model_input = torch.cat(
                [depth.unsqueeze(dim=1), curr_images, gt_img], dim=1)

            # Pass the inputs to the model
            out = model(model_input)

            ex_curr_new = calculate_T_CO_pred(
                out, ex_curr.to(device), rot_repr='SVD')

            outR = ex_curr_new[:,:3,:3].detach().clone()
            angle = angle_error(outR, ex[:,:3,:3]).mean().item()

            angle_errors.append(angle)

            # Calculate loss
            loss = compute_disentangled_ADD_L1_loss(
                ex_curr_new.to(device), ex.to(device), verts)

            ex_curr = ex_curr_new.detach()

            # Backward
            loss.backward()
            opt.step()

            epoch_loss.append(loss.item())
    return epoch_loss, angle_errors

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
        for it in range(iterations):
            opt.zero_grad()

            ex = ex.to(device)
            verts = verts.to(device)
            gt_img = images.to(device, dtype=torch.float)

            if(it == 0):
                ex_curr = ex_curr.to(device)

            if(it > 0):
                curr_images = render_to_batch_parallel(cad_id.cpu(), class_id, ex_curr.detach().cpu().numpy(), train = True, img_res = 320)
                ex_curr = ex_curr.to(device)

            curr_images = curr_images.to(device, dtype=torch.float)


            depth = depth.to(device)
            # ALL TO GPU
            

            # Stack the inputs: image, depth, and initial guess
            model_input = torch.cat(
                [depth.unsqueeze(dim=1), curr_images, gt_img], dim=1)

            # Pass the inputs to the mo del
            out = model(model_input)

            ex_curr_new = calculate_T_CO_pred(
                out, ex_curr.to(device), rot_repr='SVD')

            # Calculate loss
            loss = compute_disentangled_ADD_L1_loss(
                ex_curr_new.to(device), ex.to(device), verts)

            
            ex_curr = ex_curr_new.detach()
            # Backward
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

    #print(ex)
    #print(ex_curr_new)
    #print(ex_curr_init)
    #print(loss.item())
    #sanity(gt_img.detach().cpu().numpy(), curr_images.detach().cpu().numpy(), ex.detach().cpu(
    #).numpy(), ex_curr.detach().cpu().numpy(), ex_curr_new.detach().cpu().numpy(), cad_id, class_id)

    return epoch_loss




def test_net(model, dl_eval, device, iterations=1, lossfunc=None):

    epoch_loss = []
    model.eval()
    angle_errors = []
    with torch.no_grad():
        for index, (images, curr_images, depth, verts, ex, ex_curr, class_id, cad_id) in enumerate(dl_eval):
            gt_img = images.to(device, dtype=torch.float)
            curr_images = curr_images.to(device, dtype=torch.float)

            verts = verts.to(device)
            ex = ex.to(device)
            ex_curr = ex_curr.to(device)
            depth = depth.to(device)

            model_input = torch.cat(
                [gt_img, depth.unsqueeze(dim=1), curr_images], dim=1)

            out = model(model_input)

            ex_curr_new = calculate_T_CO_pred(
                out, ex_curr.to(device), rot_repr='SVD')


            loss = compute_disentangled_ADD_L1_loss(
                ex_curr_new.to(device), ex.to(device), verts)

            outR = ex_curr_new[:,:3,:3].detach().clone()
            angle = angle_error(outR, ex[:,:3,:3]).mean().item()

            angle_errors.append(angle)

            epoch_loss.append(loss.item())

  

    return epoch_loss, angle_errors


# Brief setup
rot_rep = 'SVD'
rot_dim = 9

num_classes = 1
batch_size = 64
pregen = True
if(pregen):
    dataset_dir = 'dataset_iteration_2/'
else:
    dataset_dir = 'data/'

dataset = ''
epochs = 300
drop_epochs = []
save_interval = 1
model_name = 'resnetrs101'
ngpu = 2
lr = 0.05


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


classes = ['rendered']
if(pregen):
    dl_train, dl_eval = get_6D_loader_pregen(dataset, batch_size, dataset_dir, shuffle = True)
else:
    dl_train, dl_eval = get_6D_loader(
        dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False)

# model = ResnetRS.create_pretrained(
#    model_name, in_ch=6, num_classes=num_classes)
resnet101_7_channel = Resnet(101, 7)

# use resnet34_4_channels(False) to get a non pretrained model
model = resnet101_7_channel(True)
model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)

opt = torch.optim.SGD(model.parameters(), lr=lr)

resume = False
curr_epoch = 0
LOAD_PATH = 'logs/run014/saved_models/resnetrs101_state_dict_199.pkl'
if(resume):
    NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
    LOAD_PATH = os.path.join(LOAD_PATH, NAME)
    model, opt, epoch = load_network(
        'logs/run009/saved_models/resnetrs101_state_dict_40.pkl', model, opt, model_name, rot_dim, num_classes)
    for g in opt.param_groups:
        g['lr'] = lr
    print("Resuming training from epoch: ", epoch)



writer_train = SummaryWriter(
    log_dir=os.path.join(SAVE_PATH, 'train'), comment=f"_{model_name}_{opt.__class__.__name__}_{lr}_train")

for e in range(curr_epoch, epochs):
    # hva må gjøres hver epoke?
    verbose = e % int(save_interval) == 0 or e == (epochs - 1)

    epoch_time = time.time()

   
    if(e < 100):
        train_loss, train_angle_errors = train_net_pregren(model, opt, dl_train, device, writer_train,
                            iterations=1, lossfunc=lossfunc)
    else:
        train_loss, train_angle_errors = train_net_pregren(model, opt, dl_train, device, writer_train,
                            iterations=2, lossfunc=lossfunc)

    val_loss, val_angle_errors = test_net(model, dl_eval, device,
                        iterations=1, lossfunc=lossfunc)

    average_train_loss = np.mean(train_loss)
    average_eval_loss = np.mean(val_loss)
    average_train_angle_error = np.mean(train_angle_errors)
    average_eval_angle_error = np.mean(val_angle_errors)
    median_train_angle_error = np.median(train_angle_errors)
    median_eval_angle_error = np.median(val_angle_errors)

    #average_eval_loss = (sum(val_loss) / len(val_loss))

    writer_train.add_scalar('Loss/train', average_train_loss, e)
    writer_train.add_scalar('Loss/test', average_eval_loss, e)
    writer_train.add_scalar('MeanAngleError/train', average_train_angle_error, e)
    writer_train.add_scalar('MeanAngleError/test', average_eval_angle_error, e)
    writer_train.add_scalar('MedianAngleError/train', median_train_angle_error, e)
    writer_train.add_scalar('MedianAngleError/test', median_eval_angle_error, e)

    #writer_train.add_scalar('Loss/test', average_eval_loss, e)

    print(e, "/", epochs, ": ", "Training loss: ", average_train_loss,
          "Validation loss: ", average_eval_loss, "time: ", epoch_time)

    if(verbose):
        save_network(e, model, opt, model_name, MODEL_SAVE_PATH)
