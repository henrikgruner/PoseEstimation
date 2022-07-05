### 3D Pose Estimation tasks
3D/6D-Pose estimation with deep learning using different rotation representation methods with special emphasis on singular value decomposition. Each folder of interest has its own README.

## Comparison test
The comparison test is to find out which rotation representation achieves the lowest mean angle error for an autoencoder structure for rotation matrices. Output from neural networks are usually on real Euclidean space, whereas rotation matrices are on the special orthogonal space (SO(3)). Special orthogonal refers to matrices $A$ with $AA^T = I$ and $\mathrm{det} A = +1$. One can try to directly regress and hope that a neural networks learns these constraints, but experiments shows it will do a poor job. Hence a mapping function is typically used, e.g., Euler angles or quaternions. Recent works has shown that other orthogonalization methods such as Gram-Schmidth and symmetric orthogonalization via Singular Value Decomposition (SVD) achievies a much greater perfomance than the traditional ones. This is tested with the comparison test, which demonstrates that SVD is the superior method (given Frobenius norm as loss function). Check the comparison folder for the code and results


## UPNA Head Pose estimation
The UPNA Head Pose estimation test is to see if a model is able to regress the rotation matrices on real humans heads. The model managed to do better than the origianl author quoting a mean angle error of 8.3, whereas my model managed 5.7 degrees. Check the UPNA folder for details


## 3D Pose Estimation from 2D images
Estimating the 3D pose of an object given an 2D image is an important task in robotics and has achieved great progress the last years. I tested the state-of-the-art models such as ResNet-RS101 with SVD mapping and Gram-Schmidt 6D mapping an achieved significant better results than the comparison. Check the 3D-Pose folder for details.

## Iterative Pose Refinement
As an ultimate experiment, a refinement process inspired by DeepIM and CosyPose was tested which sought out to increase the original guesses made by a model. Expanding to 6D, translation was included. It is possible to keep this fixed, by setting the translation equal to the ground truth. Check the Iterative folder for details

## Loss function
All models in all experiments uses the Frobenius norm
$$\mathcal{L} = ||\hat{R}-R||_F^2$$
