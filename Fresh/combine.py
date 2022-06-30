import matplotlib.pyplot as plt

def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:, :, 0] = gs1
    img[:, :, 1] = gs2
    img[:, :, 2] = 255
    return img