# 3D pose estimation

## Data set
The data set used was the ModelNet10-SO(3) created by [1]. The data set can be downloaded from https://drive.google.com/file/d/17GLZbNTDq8B_MOgrV1TiJPoqcm_oQ_mK/view?usp=sharing, with the corresponding github repo https://github.com/leoshine/Spherical_Regression#ModelNet10-SO3-Dataset. The format is stored on a lmdb format, and for my usage was turned into pickle format. To do this, run

```
python3 dataset.py
```

which will generate pickle classes. Note that this takes upwards of 200-300 GB of memory, hence one can also use [1] method. However this requires changing the code in my project. 

## Network overview
The network used was a ResNet-RS101 [2], and an overview is illustrated below, with SVD as the rotation representation. 

![overview](https://github.com/henrikgruner/PoseEstimation/blob/master/3D-Pose/git_imgs/overview.png)


## results
The results from the model was:
![overview](https://github.com/henrikgruner/PoseEstimation/blob/master/3D-Pose/git_imgs/results.png)

A few examples from the test set
![overview](https://github.com/henrikgruner/PoseEstimation/blob/master/3D-Pose/git_imgs/examples.png)

An entire batch from a test set (100 examples)
![overview](https://github.com/henrikgruner/PoseEstimation/blob/master/3D-Pose/git_imgs/batch.png)


## References
[1] Liao, S., Gavves, E., & Snoek, C. G. Supplementary Material Spherical Regression: Learning Viewpoints, Surface Normals and 3D Rotations on n-Spheres.

[2] Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T. Y., ... & Zoph, B. (2021). Revisiting resnets: Improved training and scaling strategies. Advances in Neural Information Processing Systems, 34, 22614-22627.
