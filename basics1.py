import torch

# Scalar: 0D tensor
scalar = torch.tensor(5)
print(f"Scalar: {scalar} \n " \
      f"Shape: {scalar.shape}\n\n")

# Vector: 1D tensor
vector = torch.tensor([1, 2, 3])
print(f"Vector: {vector} \n " \
      f"Shape: {vector.shape}\n\n")

# Matrix: 2D tensor
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])
print(f"Matrix: {matrix} \n " \
      f"Shape: {matrix.shape}\n\n")

# 3D Tensor
tensor_3d = torch.randn(1, 3, 4)
print(f"Tensor: {tensor_3d} \n " \
      f"Shape: {tensor_3d.shape}")

# Create tensor from a Python list or number.
# tensor = torch.tensor()

# Create tensor filled with zeros.
# zeros = torch.zeros()

# Create tensor filled with ones.
# ones = torch.ones()

# Create tensor filled with random numbers between 0 and 1.
# radn = torch.rand()

# Different type of 3D tensor
tensor_3_1_d = torch.rand((2, 3, 4))
print(f"Different type of Tensor: {tensor_3_1_d}\n\n" \
      f"Shape: {tensor_3_1_d.shape}")
