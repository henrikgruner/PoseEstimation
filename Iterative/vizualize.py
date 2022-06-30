import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append('../data')
sys.path.append('..')
sys.path.append('logs')
from dataset import get_6D_loader, get_6D_eval_loader,get_6D_loader_pregen
import os
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
sys.path.append('../modelnet/ModelNet40')
sys.path.append('../data')
sys.path.append('../CosyPose')
from render_utility import *
import multichannel_resnet
from multichannel_resnet import get_arch as Resnet


def plot(imgs, titles, save_title, angles):
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = len(imgs)
    
    for i, img in enumerate(imgs):
        print(img.shape)    
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(titles[i]+str(angles[i]))

    plt.savefig(os.path.join('illustrations',save_title))
    plt.close()


def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:, :, 0] = gs1
    img[:, :, 1] = gs2
    img[:, :, 2] = 255
    return img



def vizualiser(model, dl_train, dl_eval, out_dim =12, num_samples = 50, fix = True):
    model.eval()
    print('gg')
    device = 'cuda'
    with torch.no_grad():
        for index, (images, curr_images,render_img, depth, verts, ex, ex_curr, mod_ex, class_id, cad_id) in enumerate(dl_eval):
            angles = ['','']
            ang1 = []
            ang2 = []
            ang3 = []

            if(index > num_samples):
                exit()
            ex_test = ex_curr.detach().cpu().clone()
            for it in range(2):
                

                ex = ex.to(device)
                verts = verts.to(device)
                gt_img = images.to(device, dtype=torch.float)

                if(it == 0):
                    current_images = curr_images
                    ex_current = ex_curr.to(device)

                if(it > 0):
                    current_images = render_img
                    
                    ex_current = mod_ex.to(device)

                current_images = current_images.to(device, dtype=torch.float)
                depth = depth.to(device)


                model_input = torch.cat(
                    [depth.unsqueeze(dim=1), current_images, gt_img], dim=1)

                out = model(model_input)

                ex_curr_new = calculate_T_CO_pred(
                    out, ex_current.to(device), rot_repr='SVD')

                outR = ex_curr_new[:,:3,:3].detach().clone()

                angle = round(angle_error(outR, ex[:,:3,:3]).mean().item(),1)

                angles.append(angle)



            if(fix):
                ex_current[:,0,3] = ex[:,0,3].detach().cpu().clone()
                ex_current[:,1,3] = ex[:,1,3].detach().cpu().clone()
                ex_current[:,2,3] = ex[:,2,3].detach().cpu().clone()

                ex_curr_new[:,0,3] = ex[:,0,3].detach().cpu().clone()
                ex_curr_new[:,1,3] = ex[:,1,3].detach().cpu().clone()
                ex_curr_new[:,2,3] = ex[:,2,3].detach().cpu().clone()
                ex_test[:,0,3] = ex[:,0,3].detach().cpu().clone()
                ex_test[:,1,3] = ex[:,1,3].detach().cpu().clone()
                ex_test[:,2,3] = ex[:,2,3].detach().cpu().clone()

                ang1.append(angle_error(ex_current[:,:3,:3].detach().cpu().clone(), ex[:,:3,:3].detach().cpu().clone()))
                ang2.append(angle_error(ex_test[:,:3,:3].detach().cpu().clone(), ex[:,:3,:3].detach().cpu().clone()))
                ang3.append(angle_error(ex_curr_new[:,:3,:3].detach().cpu().clone(), ex[:,:3,:3].detach().cpu().clone()))


            print(np.mean(ang1),np.mean(ang2),np.mean(ang3))
'''
            images = images.detach().cpu().numpy()[0].transpose(1,2,0)
            curr_images = curr_images.detach().cpu().numpy()[0].transpose(1,2,0)
            render_img = render_img.detach().cpu().numpy()[0].transpose(1,2,0)

            mod_img =  render_to_batch(cad_id.cpu(), class_id, ex_test.detach().cpu().numpy(), train = False, img_res = 320)
            mod_img = mod_img.detach().cpu().numpy()[0].transpose(1,2,0) 

            curr_images = render_to_batch(cad_id.cpu(), class_id, ex_current.detach().cpu().numpy(), train = False, img_res = 320)
            curr_images = curr_images.detach().cpu().numpy()[0].transpose(1,2,0)  

            render_2 = render_to_batch(cad_id.cpu(), class_id, ex_curr_new.detach().cpu().numpy(), train = False, img_res = 320)
            render_2 = render_2.detach().cpu().numpy()[0].transpose(1,2,0)

            img = [images, curr_images, render_img, render_2]
            combine_init = [images, combine_imgs(images, curr_images), combine_imgs(images, mod_img), combine_imgs(images, render_2)]

            plot(img, titles =['Gt', 'guess 1:', 'guess 2:', 'guess 3:'],save_title = str(index), angles = angles)
            plot(combine_init, titles =['Gt', 'guess 1:', 'guess 2:', 'guess 3:'],save_title = str(index)+'combine', angles = angles)
'''


def load_network_2(path, model):
    
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])

    epoch = modelcheckpoint['epoch']

    return model




def main():
    print('maaaain')
    batch_size = 1
    pregen = True
    dataset = ''

    if(pregen):
       dataset_dir = 'dataset_iteration_2/'
    else:
        dataset_dir = 'data/'

    model_name = 'resnetrs101'

    classes = ['homo']
    if(pregen):
        dl_train = get_6D_loader_pregen(dataset, batch_size, dataset_dir, shuffle = True)
    else:
        dl_train, dl_eval = get_6D_loader(
            dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False)

    device = torch.device("cuda")
    devices = [d for d in range(torch.cuda.device_count())]
    resnet101_7_channel = Resnet(101, 7)
    model = resnet101_7_channel(True)   
    model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)
    LOAD_PATH = 'logs/run067/saved_models/resnetrs101_state_dict_221.pkl'

    model = load_network_2(LOAD_PATH, model)
    vizualiser(model, dl_train, dl_train, out_dim = 12, num_samples = 50)

if __name__ == '__main__':
    main()