### ModelNet10-SO(3)

ModelNet10 [2] is a collection of households objects in 3D. For application of 3D pose estimation from 2D images, the objects needs to be rendered(i.e., take an image). For an ease of comparison between data sets (rendering has many parameters), I used ModelNet10-SO3 created by [1]. Each object was rotated randomly 100 times, and an image is rendered from each. The annontations given are the rotation matrix, intrinsic matrix, in addition to the cad id and class index. For the test set, only 10 are created. 

To achieve more flexibility (use just one class instead of all 10 or limited amounts) the data set was extracted and turned into Pickle format. This requires a lot of memory (2-300 GB). To do this, run


```
python3 generate.py
```

which will extract each class and store them seperatly (train/test) in /datasetSO3. Note that this takes upwards of 200-300 GB of memory, hence one can also use [1] method. However this requires changing the code in my project. 

[1] Liao, S., Gavves, E., & Snoek, C. G. Supplementary Material Spherical Regression: Learning Viewpoints, Surface Normals and 3D Rotations on n-Spheres.

[2] Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., & Xiao, J. (2015). 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1912-1920).