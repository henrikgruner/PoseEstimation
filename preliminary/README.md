## Comparison of rotation representations with rotation matrix regression
The code is a simple autoencoder structure which takes in a randomly rotated coordinate system in 3D, i.e., a rotation matrix working as a frame. The target is the rotation matrix, hence the structure is essentially an autoencoder. Rotation matrices belong in the special orthogonal space SO(3), whereas the output of a neural network lies on the real Euclidean space $\mathbb{R}^3$. These two spaces are not topological
homeomorphic, hence a mapping function is needed. The code in this section is meant to be a comparison between mapping functions. The code uses a simple feed-forward network (MLP) with two hidden layers of 128 and 64 units.

### Mapping functions:

1. Euler angles: $f \in \mathbb{R}^3$
2. Quaternions: $f \in \mathbb{R}^4$
3. Gram Schmidt orthogonalizaion in 5D: $f\in \mathbb{R}^5$
4. Gram Schmidt orthogonalization in 6D: $f \in \mathbb{R}^6$
5. Symmetric orthogonalization via SVD: $f \in \mathbb{R}^9$
6. Direct regression without orthogonalization: $f \in \mathbb{R}^9$. This is not a true rotation representation, as there is no guarantee it will be orthogonal. It is included nonetheless to see how well the network is able to perform without a parametrization function. The output is simply reshaped into $3 \times 3$

