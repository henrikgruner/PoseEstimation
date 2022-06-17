import pyrender
import sys
sys.path.append('../modelnet/ModelNet40')
sys.path.append('../data')
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_6D_loader, get_modelnet_loaders
import os
import pyrender
import trimesh as tm
from render_utility import *
from loss import *
os.environ['PYOPENGL_PLATFORM'] = 'egl'
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


from skimage import data, color, io, img_as_float
import numpy as np
import matplotlib.pyplot as plt


def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:, :, 0] = gs1
    img[:, :, 1] = gs2
    img[:, :, 2] = 255
    return img


if __name__ == '__main__':
    classes = ['chair']
    batch_size = 1
    dataset_dir = 'data/'
    dataset = ''
    dl_train, dl_eval = get_6D_loader(
        dataset, batch_size, dataset_dir, classes)

    print(len(dl_train))
    print(len(dl_eval))
    '''
    i = 0
    for img, init_image, depth, verts, ex, init_ex, class_id, cad_id in dl_eval:
        i += 1
        if(i > 10):
            exit()
        img = img.to('cpu').detach().numpy()
        img = img[0].transpose(1, 2, 0)

        init_img = init_image.to('cpu').detach().numpy()
        init_img = init_img[0].transpose(1, 2, 0)

        combined_img = combine_imgs(img, init_img)
        plot(img, init_img, combined_img, str(i))
        # img_rerender, mesh_points = render_to_batch(
        #    cad_id, c_id, t, batch_size)

        loss_Cosy = compute_ADD_L1_loss(ex, init_ex, verts)
        dis_loss = compute_disentangled_ADD_L1_loss(ex, init_ex, verts)
        print(i)
        print(ex)
        print(init_ex)
        print(loss_Cosy)
        print(dis_loss)
        print('\n\n')


    R = np.array([[0.3598, 0.8291, -0.4280, -0.0698],
                  [0.0176, 0.4526, 0.8915, 0.1329],
                  [0.9329, -0.3283, 0.1483, -2.5796],
                  [0.0000, 0.0000, 0.0000, 1.0000]])

    plt.imshow(img_rerender)
    plt.savefig("render2")
    plt.close()

    img = img.to('cpu').detach().numpy()
    img = img[0].transpose(1, 2, 0)
    plt.imshow(img)
    plt.savefig("main1")
    plt.close()

    combine = combine_imgs(img, img_rerender)
    plt.imshow(combine)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("combo")
    plt.close()
    
    
    print(ex)
    print(t)
    R = np.array([[0.3598, 0.8291, -0.4280, -0.0698],
         [0.0176, 0.4526, 0.8915, 0.1329],
         [0.9329, -0.3283, 0.1483, -2.5796],
         [0.0000, 0.0000, 0.0000, 1.0000]])
    print(R)
    print(R * (1 + np.random.uniform(0, 0.1)))
    gg = [9.81, 79.9, 16.89, 15.39, 53.08, 27.69, 40.41, 52.4, 23.4, 93.33]
    med = [5.35, 34.39, 5.25, 6.68, 20.03, 7.33, 9.3, 18.52, 5.78, 97.17]
    print(np.median(med))
    '''
