import torch
import ModelNetSO3
import pandas as pd


def get_modelnet_loader(batch_size, train_all, dataset_dir=''):
    dataset = ModelNetSO3.ModelNetSo3(dataset_dir)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval


def main(classes, filename, train=True):

    dataset = ModelNetSO3.ModelNetSo3('')

    if train:
        data = dataset.get_train()
    else:
        data = dataset.get_eval()

    gg = 0
    curr_class = classes[gg]

    image, extrinsic, class_idx, hard, intrinsic, cad_idx = [], [], [], [], [], []

    for j in range(0, len(data)):
        i = data[j]
        if str(i[2]) in curr_class:
            image.append(i[0])
            extrinsic.append(i[1])
            class_idx.append(i[2])
            intrinsic.append(i[3])
            cad_idx.append(i[4])
        else:
            df = pd.DataFrame()
            df['image'] = image
            df['extrinsic'] = extrinsic
            df['class_idx'] = class_idx
            df['intrinsic'] = intrinsic
            df['cad_idx'] = cad_idx
            df.to_pickle('datasetSO3/' + str(curr_class[0]) + filename)

            image, extrinsic, class_idx, hard, intrinsic, cad_idx = [], [], [], [], [], []
            gg += 1
            try:
                curr_class = classes[gg]
            except Exception:
                break

    df = pd.DataFrame()
    df['image'] = image
    df['extrinsic'] = extrinsic
    df['class_idx'] = class_idx
    df['intrinsic'] = intrinsic
    df['cad_idx'] = cad_idx
    df.to_pickle('datasetSO3/' + str(curr_class[0]) + filename)

   # dataset = ModelNetSO3.ModelNetSo3(dataset_dir)data,_ = get_modelnet_loaders(8, True)


if __name__ == '__main__':
    #classes = [['bathtub'], ['bed'], ['chair'], ['desk'], ['dresser'], [
    #    'monitor'], ['night_stand'], ['sofa'], ['table'], ['toilet']]
    classes = [['bathtub']]
    filename_train = '_train.pkl'
    filename_eval = '_eval.pkl'
    main(classes, filename_train, True)
    main(classes, filename_eval, False)
