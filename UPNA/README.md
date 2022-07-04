## UPNA head pose estimation
To run the code download the data set from http://www.unavarra.es/gi4e/databases/hpdb and put it in the datasets folder. To preprocess it run
```
python3 UPNA.py
```
And the dataset will be created as in [2]. The code for the preprocessing is taken from https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions,and will create the same train/test sets for a fair comparison of results. To run the model, use
```
python3 main.py
```

Automatically a folder will be created which saves the model every fifth epoch, and logs the results with tensorboard. 


My best result was 5.7 degrees, whilst [2] achieves 4.3 degrees. The original authors of UPNA [1] is only able to achieve 8.3 degrees. 

## References
<a id="1">[1]</a> 
Ariz, M., Bengoechea, J. J., Villanueva, A., & Cabeza, R. (2016).
A novel 2D/3D database with automatic face annotation for head tracking and pose estimation.
Computer Vision and Image Understanding, 148, 201-210.
<a id="2">[2]</a> 
Mohlin, D., Sullivan, J., & Bianchi, G. (2020).
Probabilistic orientation estimation with matrix fisher distributions.
Advances in Neural Information Processing Systems, 33, 4884-4893.

