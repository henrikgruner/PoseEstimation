# 3D pose estimation

## Data set
The data set used was the ModelNet10-SO(3) created by [1]. The data set can be downloaded from https://drive.google.com/file/d/17GLZbNTDq8B_MOgrV1TiJPoqcm_oQ_mK/view?usp=sharing, with the corresponding github repo https://github.com/leoshine/Spherical_Regression#ModelNet10-SO3-Dataset. The format is stored on a lmdb format, and for my usage was turned into pickle format. Put the downloaded files into the dataset folder. Further instructions can be found there.


## To run
The configs are stored in configs. An example config can be found as example.yaml. Note that I ran with 4 A100 40GB gpu's, hence the batch size should likely be modified for usage with fewer gpus. The training time is long, and might take 1-2 days, depending on the set up. 
```
python3 main.py -c [config_name] -dir [config_directory]
```
The default config is example.yaml and the default directory is configs. All the parameters will be printed and the progress will be printed every epoch. 

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

These rendering were created with the original ModelNet10, and not ModelNet10-SO3, as rendering is required. Feel free to contact me if you have any questions regarding the illustrations of the estimations. 

## References
[1] Liao, S., Gavves, E., & Snoek, C. G. Supplementary Material Spherical Regression: Learning Viewpoints, Surface Normals and 3D Rotations on n-Spheres.

[2] Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T. Y., ... & Zoph, B. (2021). Revisiting resnets: Improved training and scaling strategies. Advances in Neural Information Processing Systems, 34, 22614-22627.
