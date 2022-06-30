import pyrender
import sys
sys.path.append('../modelnet/ModelNet40')
sys.path.append('../data')
sys.path.append('../CosyPose')

#from model import ResnetRS

import matplotlib.pyplot as plt
import numpy as np
from dataset import get_6D_loader, get_modelnet_loaders
import os
import pyrender
import trimesh as tm
from render_utility import *
from loss import *
from multichannel_resnet import get_arch as Resnet

import os


# TODO: Render ModelNet10

# 1. Datasettet må normaliseres: z = 5.
# 2. Rendering med random T_CO
# 3. Få tak i R fra nettverket
# 4. Få tak i translasjon fra (?)
# 5. Kombiner bildene
# 6. Kjør ResNet-RS


def render_image(path='toilet.off'):
    raise NotImplementedError


def get_small_change_T(T, mac, mtc):
    '''
    Function takes a transformation matrix
    and alters it randomly in the ranges given
    Input:
    - T - transform matrix
    - mac - Maximum angle change (degrees)
    - mtc - Maximum translation change (meters)
    Output:
    - T_random = slightly altered T
    '''
    rand = np.random.uniform
    change = rand(-mac, mac)
    x_change, y_change, z_change = rand(-mtc, mtc, size=3)

    R = sm.SO3.AngleAxis(change, get_random_unit_axis(), unit='deg')
    transl = np.array([x_change, y_change, z_change])
    T_new = sm.SE3.Rt(R, transl)

    return T_new * T


def class_mapping(c_id=None, c_str=None):
    classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
               'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    mapping = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4,
               'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9}
    if(c_id is None):
        return mapping[c_str]
    else:
        return classes[c_id]


def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:, :, 0] = gs1
    img[:, :, 1] = gs2
    img[:, :, 2] = 255
    return img


def plot(img, init_img, combined_img, title):
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 3
    # showing image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Truth")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(init_img)
    plt.axis('off')
    plt.title("Prediction")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(combined_img)
    plt.axis('off')
    plt.title("Both")

    plt.savefig('all_figs' + str(title))
    plt.close()


def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:, :, 0] = gs1
    img[:, :, 1] = gs2
    img[:, :, 2] = 255
    return img


def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch


def test_model():
    Fresh = False
    if(Fresh):
        path = '../Fresh/logs2/run020/saved_models/resnetrs101_state_dict_530.pkl'
        model_name = 'resnetrs101'
        num_classes = 1
        model = ResnetRS.create_pretrained(
            model_name, in_ch=3, num_classes=num_classes)
    else:
        path = 'logs/run014/saved_models/resnetrs101_state_dict_199.pkl'
        resnet101_7_channel = Resnet(101, 7)
        model = resnet101_7_channel(True)

    device = torch.device("cuda")
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]

    print("cuda: ", torch.cuda.is_available())
    print("count: ", torch.cuda.device_count())
    print("names: ", device_names)
    print("device ids:", devices)

    model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0)

    import matplotlib.pyplot as plt

    classes = ['chair', 'toilet']
    batch_size = 1

    dataset_dir = 'data/'
    dataset = ''
    dl_train, dl_eval = get_6D_loader(
        dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False)

    i = 0
    model.eval()
    with torch.no_grad():
        for img, init_img, depth, verts, ex, ex_curr, class_id, cad_id in dl_train:
            i += 1
            if(i > 100):
                exit()
            verts = verts.to(device)
            ex = ex.to(device)
            ex_curr = ex_curr.to(device)
            depth = depth.to(device)
            img = img.to(device, dtype=torch.float)

            init_img = init_img.to(device, dtype=torch.float)

            if(Fresh):

                out = model(img)
                out_R = out[:, 0:9]
                out_t = out[:, 9:12]
                x = out_t[:, 0]
                y = out_t[:, 1]
                z = out_t[:, 2]
                R = symmetric_orthogonalization(out_R)

                T = torch.zeros((img.shape[0], 4, 4))
                T[:, :3, :3] = R
                T[:, 0, 3] = x
                T[:, 1, 3] = y
                T[:, 2, 3] = -2.5
                T[:, 3, 3] = 1
                ex_curr_mod = T
            else:
                gt_img = img.to(device, dtype=torch.float)
                curr_images = init_img.to(device, dtype=torch.float)
                verts = verts.to(device)
                ex = ex.to(device)
                ex_curr = ex_curr.to(device)
                depth = depth.to(device)

                # Stack the inputs: image, depth, and initial guess
                model_input = torch.cat(
                    [depth.unsqueeze(dim=1), curr_images, gt_img], dim=1)

                # Pass the inputs to the model
                out = model(model_input)

                #Rotation = out[:, :9]
                #t = out[:, 9:12]
                #R = symmetric_orthogonalization(Rotation)
                #ex_curr = SE3_parameterization(R, t, ex_curr.to(device))

                # Calculate new guess for ex_current
                ex_curr_new = calculate_T_CO_pred(
                    out, ex_curr.to(device), rot_repr='SVD')
                # Få inn symmetri her a

            loss = compute_disentangled_ADD_L1_loss(
                ex_curr_new.to(device), ex.to(device), verts)

            print(ex[:, 2, 3])
            print(ex_curr_new[:, 2, 3])
            print(loss.item())

            '''
            model_img, _ = render_from_id(
                cad_id.item(), class_id, ex_curr_mod.cpu().detach().numpy()[0], train=True)

            print(model_img.shape)
            print(img.shape)

            img = img.to('cpu').detach().numpy()
            img = img[0].transpose(1, 2, 0)

            init_img = init_img.to('cpu').detach().numpy()
            init_img = init_img[0].transpose(1, 2, 0)

            combined_img = combine_imgs(img, init_img)
            combined_img2 = combine_imgs(img, model_img)

            plot(img, init_img, model_img, str(i))

            plot(img, combined_img, combined_img2, str(i + 100))


            print(i)
            print(ex)
            print(ex_curr)
            print(ex_curr_mod)
            print('\n\n')
            '''

if __name__ == '__main__':


    classes = ['chair']
    batch_size = 1

    dataset_dir = 'data/'
    dataset = ''
    dl_train, _ = get_6D_loader(
        dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False)
    print(len((dl_train)))