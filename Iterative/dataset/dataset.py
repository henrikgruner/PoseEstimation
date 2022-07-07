import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


def normalize(depth):
    '''
    Ola Alstad - Guden
    '''
    mean_val = np.mean(depth[depth > 0.01])
    std = np.std(depth[depth > 0.01])
    normdepth = np.where(depth > 0.01, (depth - mean_val) / std, 0.0)
    return normdepth


class ModelNet10SO3(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, classes, train=True):
        if(train):
            end = '_train.pkl'
        else:
            end = '_eval.pkl'
        self.df = pd.read_pickle(dataset_dir + classes[0] + end)

        for i in range(1, len(classes)):
            new_df = pd.read_pickle(dataset_dir + classes[i] + end)
            self.df = pd.concat([self.df, new_df])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = row.image.transpose(1, 2, 0)
        image = torchvision.transforms.functional.to_tensor(image)
        return image, row.extrinsic, row.class_idx,row.intrinsic, row.cad_idx
        #image = torchvision.transforms.functional.to_tensor(row.image)


class ModelNet6D_pregen(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, classes, train=True):
        if(train):
            end = '_train.pkl'
        else:
            end = '_test.pkl'

        self.df = pd.read_pickle(dataset_dir + classes[0] + end)

        for i in range(1, len(classes)):
            new_df = pd.read_pickle(dataset_dir + classes[i] + end)
            self.df = pd.concat([self.df, new_df])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = row.Images
        init_image = row.Init_Images

        norm_depth = normalize(row.Depth)

        image = torchvision.transforms.functional.to_tensor(image)
        init_image = torchvision.transforms.functional.to_tensor(init_image)
        depth = torch.tensor(norm_depth)
        ex = row.Extrinsic.astype(np.float32)
        init_ex = row.Extrinsic_init.astype(np.float32)
        verts = row.Vertices
        class_id = row.Class
        cad_id = row.Cad_id

        return image, init_image, depth, verts, ex, init_ex, class_id, cad_id

class ModelNet6D_pregen2(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, train=False):
        if(train):
            end = '_train.pkl'
        else:
            end = '_test.pkl'

        if(dataset_dir == 'dataset_iteration_2/'):
            self.df = pd.read_pickle(dataset_dir+'homo'+ end)
            #for i in range(2,17):
            #    new_df = pd.read_pickle(dataset_dir + 'rendered_'+str(i)+ end)
            #    self.df = pd.concat([self.df, new_df])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = row.Images.transpose(1,2,0)
        init_image = row.Init_Images.transpose(1,2,0)
        norm_depth = normalize(row.Init_depth)
        render_img = row.Rendered_img.transpose(1,2,0)

        image = torchvision.transforms.functional.to_tensor(image)
        init_image = torchvision.transforms.functional.to_tensor(init_image)
        render_img = torchvision.transforms.functional.to_tensor(render_img)
        depth = torch.tensor(norm_depth)
        ex = row.Extrinsic.astype(np.float32)
        mod_ex = row.Extrinsic_rendered.astype(np.float32)
        init_ex = row.Extrinsic_init.astype(np.float32)
        verts = row.Vertices
        class_id = row.Class
        cad_id = row.Cad_id

        return image, init_image,render_img, depth, verts, ex, init_ex,mod_ex, class_id, cad_id




class ModelNet6D_pregen_stripped(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, classes, train=True):
        if(train):
            end = '_train.pkl'
        else:
            end = '_test.pkl'
    
        self.df = pd.read_pickle(dataset_dir + classes[0] + end)

        for i in range(1, len(classes)):
            new_df = pd.read_pickle(dataset_dir + classes[i] + end)
            self.df = pd.concat([self.df, new_df])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = row.Images

        image = torchvision.transforms.functional.to_tensor(image)

        ex = row.Extrinsic.astype(np.float32)

        class_id = row.Class
        cad_id = row.Cad_id

        return image, ex, class_id, cad_id


class ModelNet6D(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, classes, RGBD=False, train=True):
        if(train):
            end = '_train.pkl'
        else:
            end = '_test.pkl'

        self.df = pd.read_pickle(dataset_dir + classes[0] + end)

        self.RGBD = RGBD
        for i in range(1, len(classes)):
            new_df = pd.read_pickle(dataset_dir + classes[i] + end)
            self.df = pd.concat([self.df, new_df])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = row.Images
        if(self.RGBD):
            norm_depth = normalize(row.Depth)
            image = np.concatenate((image, norm_depth), axis=2)

        #image = image.transpose(1, 2, 0)
        image = torchvision.transforms.functional.to_tensor(image)

        return image, row.Extrinsic.astype(np.float32), row.Extrinsic_init.astype(np.float32), row.Class, row.Cad_id



def get_modelnet_loaders(dataset, batch_size, dataset_dir, classes, RGBD=False):

    dataset_train = ModelNet10SO3(dataset_dir, classes, True)
    dataset_eval = ModelNet10SO3(dataset_dir, classes, False)
    # elif 'Cosy' in dataset:
    #    dataset_train = CosyModelNet(
    #        dataset_dir, classes, RGBD=False, train=True)
    # dataset_eval = CosyModelNet(
    #       dataset_dir, classes, RGBD=False, train=False)
    # else:
    #    print('Dataset does not exist')

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    return dataloader_train, dataloader_eval


def get_eval_loader(dataset, batch_size, dataset_dir, classes, RGBD=False, shuffle = False):
    if 'SO3' in dataset:
        dataset_eval = ModelNet10SO3(dataset_dir, classes, False)
    elif 'Cosy' in dataset:
        dataset_eval = CosyModelNet(
            dataset_dir, classes, RGBD=False, train=False)
    else:
        print('Dataset does not exist')
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True)
    return dataloader_eval


def get_6D_eval_loader(dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False):
    if(pre):

        dataset_eval = ModelNet6D_pregen(
            dataset_dir, classes, train=False)
    else:

        dataset_eval = ModelNet6D(
            dataset_dir, classes, RGBD=False, train=False)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True)
    return dataloader_eval


def get_6D_stripped_loader(batch_size, dataset_dir, classes):

    dataset_train = ModelNet6D_pregen_stripped(dataset_dir, classes, train=True)
    dataset_eval = ModelNet6D_pregen_stripped('../6D/data/', ['toilet'], train=False)


    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    return dataloader_train, dataloader_eval

def get_6D_loader_pregen(dataset, batch_size, dataset_dir, shuffle = True):

    dataset_train = ModelNet6D_pregen2(dataset_dir, train=False)
    #dataset_eval = ModelNet6D_pregen2('/homo', '', train=False)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    #dataloader_eval = torch.utils.data.DataLoader(
    #    dataset_eval,
    #   batch_size=batch_size,
    #    shuffle=False,
    #    num_workers=0,
    #    pin_memory=True,
    #    drop_last=True)
    return dataloader_train



def get_6D_loader(dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False, shuffle = True):
    if(pre):
        
        dataset_train = ModelNet6D_pregen(dataset_dir, classes, train=True)
        dataset_eval = ModelNet6D_pregen(dataset_dir, classes, train=False)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    return dataloader_train, dataloader_eval



def main():
    dl_eval = get_eval_loader(4, 'datasets/', ['toilet'])

    # print(len(train))
    # print(len(evals))


if __name__ == '__main__':
    main()
# plt.imshow(gg.df.iloc[1].image.transpose((-1,1,0)))
