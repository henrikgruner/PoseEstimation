## Data set 
The data set used is the ModelNet10/40 [1]. Download and unzip with with

```
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
```

The mesh needs to be rescaled and normalized. Run

```
python3 norm.py
```

Rendering is a time-consuming process, hence I created a data set beforehand with an initial estimate of the rotation matrix and a rendered image with the new estimate. This can be done for as many classes as one whish, but it is both time- and memory-consuming. For 20 samples per object each class will require around 50GB for generation of such a dataset. This can be lowered to just 2,3, or, 5 samples, but the model will overfit quickly. To create a data set use

```
python3 generate.py
```





[1] Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., & Xiao, J. (2015). 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1912-1920).
