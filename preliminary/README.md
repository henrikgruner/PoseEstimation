## Comparison of rotation representations with rotation matrix regression
The code is a simple autoencoder structure which takes in a randomly rotated coordinate system in 3D, i.e., a rotation matrix working as a frame. The target is the rotation matrix, hence the structure is essentially an autoencoder. Rotation matrices belong in the special orthogonal space, whereas the output of a neural network lies on the real Euclidean space $\mathbb{R}^3$. These two spaces are not topological
homeomorphic, hence a mapping function is needed. The code in this section is meant to be a comparison between mapping functions. The code uses a simple feed-forward network (MLP) with two hidden layers of 128 and 64 units.

### Mapping functions:
