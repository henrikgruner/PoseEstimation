import sys
sys.path.append('../modelnet/ModelNet40')
sys.path.append('../data')
sys.path.append('../Fresh')

import threading
import torchvision
import random
import trimesh as tm
from utility import *
import torch.nn as nn
from model import ResnetRS
from get_initial_estimate import get_Rinit_from_net, load_network
from dataset import get_6D_loader, get_6D_eval_loader
import torch
import matplotlib.pyplot as plt
import pyrender
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'



# TODO: Render ModelNet10

# 1. Datasettet må normaliseres: z = 5. Utsatt
# 2. Rendering med random T_CO

# Pseudokode treningsløkke


# 3. Få tak i R fra nettverket
# 4. Få tak i translasjon fra (?)
# 5. Kombiner bildene
# 6. Kjør ResNet-RS
# 7. For loss trenger man mesh.

# Not viable option yet to use own, due to translation faults.
# Dataset har: (Ikke depth)
# image, Extrinsic, Extrinsic_init, Class, Cad_id

def get_scene(mesh, ex, flen, img_res, sw):
    ex = np.linalg.inv(ex)
    scene = pyrender.Scene()
    scene.bg_color = (0, 0, 0)
    scene.add(mesh)

    ppm = sw / img_res
    fx = fy = flen / ppm
    vx = vy = img_res / 2

    camera = pyrender.IntrinsicsCamera(fx, fy, vx, vy)
    light = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                               innerConeAngle=np.pi / 8.0,
                               outerConeAngle=np.pi / 3.0)
    scene.add(light, pose=ex)
    scene.add(camera, pose=ex)

    return scene


def class_mapping(c_id=None, c_str=None):
    classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
               'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    mapping = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4,
               'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9}
    if(c_id is None):
        return mapping[c_str]
    else:
        return classes[c_id]


def sample_vertices_as_tensor(verts, num_verts=1000):
    '''
    Taken from Ola Alstad
    https://github.com/olaals/end-to-end-RGB-pose-estimation-baseline
    '''
    sampled_verts = []
    for i in range(num_verts):
        vert = np.array(random.choice(verts))
        sampled_verts.append(vert)
    sampled_verts = np.array(sampled_verts, dtype=np.float32)
    tensor_verts = torch.tensor(sampled_verts)
    return tensor_verts


def render_to_batch(cad_ids, classes_id, ex_list, train=True, img_res=244):
    batch_size = cad_ids.shape[0]
    imgs = []
    points = []

    for i, (cad_id, class_id, ex) in enumerate(list(zip(cad_ids, classes_id, ex_list))):
        img, mesh_points = render_from_id(cad_id.item(), class_id, ex)
        img = torchvision.transforms.functional.to_tensor(img)
        imgs.append(img)
        points.append(mesh_points)
    return torch.stack(imgs), torch.stack(points)


def worker_render_from_id(i, cad_id, class_id, ex_curr, imgs, points, dataset='ModelNet40/NormalizedModelNet40', train=True):
    #print("worker", i)
    img, mesh_points = render_from_id(cad_id, class_id, ex_curr)

    imgs[i, :, :, :] = torchvision.transforms.functional.to_tensor(img)
    points[i, :, :] = mesh_points.clone().detach()
    #print("worker", i, "finished")


def render_to_batch_parallel(cad_ids, classes_id, ex_list, train=True, img_res=320):
    batch_size = cad_ids.shape[0]
    imgs = torch.zeros((batch_size, 3, img_res, img_res), dtype=torch.float32)
    points = torch.zeros((batch_size, 750, 3), dtype=torch.float32)
    threads = []

    for i, (cad_id, class_id, ex) in enumerate(list(zip(cad_ids, classes_id, ex_list))):
        t = threading.Thread(target=worker_render_from_id, args=(
            i, cad_id.item(), class_id, ex, imgs, points))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    return imgs, points

# render_to_batch -> render_from_id


def render_from_id(cad_id, class_id, ex_curr, dataset='ModelNet40/NormalizedModelNet40', train=True):

    if(train):
        folder = 'train'
    else:
        folder = 'test'
    class_str = class_mapping(c_id=class_id, c_str=None)
    filename = class_str + '_' + str(cad_id).zfill(4) + '.ply'

    path = os.path.join(dataset, class_str, folder, filename)

    mesh_org = tm.load(path)
    vertices = mesh_org.vertices
    mesh = pyrender.Mesh.from_trimesh(mesh_org)

    mesh_points = sample_vertices_as_tensor(vertices)

    img = render_image(mesh, ex_curr)

    return img, mesh_points


def render_image(mesh, ex, flen=50, sw=36, img_res=320):

    scene = get_scene(mesh, ex, flen, img_res, sw)
    r = pyrender.OffscreenRenderer(
        viewport_width=img_res, viewport_height=img_res)
    color, depth = r.render(scene)
    color = color / 255
    r.delete()
    return color


def get_T_init_from_network(model, img, ex_init):
    R_init = get_Rinit_from_net(model, img, 'cuda')
    return R_init


def get_mesh_from_id(cad_id, class_id):
    raise NotImplementedError


if __name__ == '__main__':
    classes = ['toilet']
    batch_size = 8
    dataset_dir = 'dataset/'
    dataset = ''

    model_name = 'resnetrs101'

    device = torch.device("cuda" if(
        torch.cuda.is_available()) else "cpu")
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]

    model = ResnetRS.create_pretrained(
        model_name, in_ch=3, num_classes=1)

    model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0)
    path = '../Fresh/logs/run022/saved_models/resnetrs101_state_dict_28.pkl'
    model, opt, epoch = load_network(
        path, model, opt, model_name)

    dl_eval = get_6D_eval_loader(
        dataset, batch_size, dataset_dir, classes, RGBD=False)

    img, ex, ex_init, _, _ = next(iter(dl_eval))
    R = get_T_init_from_network(model, img, ex_init).to(device).float()
    R_real = ex[:, :3, :3].to(device).float()
    # print(R)
    # print(R_real)
    #print(angle_error(R, R_real))
    #print(angle_error(R_real, R))
    #img = img.to('cpu').detach().numpy()
    #img = img[0].transpose(1, 2, 0)
    # plt.imshow(img)
    # plt.savefig("siuuu")
    # plt.close()
