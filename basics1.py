import pandas as pd
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

a = torch.tensor([1, 3, 5])
b = torch.tensor([2, 4, 6])

# Element wise addition
add_result = a + b
print(f"Element wise addition:\n{add_result}")

# Element wise multiplication
mult_result = a * b
print(f"Element wise multiplication:\n{mult_result}")

# Dot product
dot_result = torch.matmul(a, b)
print(f"Dot product:\n{dot_result}")

# Pytorch also supports broadcasting:
# Allows tensors with different shapes to be combined automatically during operations
x = torch.tensor([[1, 3, 5], [4, 5, 6]])
bias = torch.tensor([2, 3, 5])

print(f"Broadcasting result: {x * bias}")

filepath = './Dataset/Concrete_Data.csv'
df = pd.read_csv(filepath)

# Calling the columns of the dataframe to a list
cols = df.columns.tolist()

# Checking for missing values
missing = df.isnull().sum()

# Inputs: all columns except the last one
inputs = df.iloc[:, :-1]

# Target: only the last column
target = df.iloc[:, -1]

# Convert inputs and targets to PyTorch tensors
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32)
target_tensor = torch.tensor(target.values, dtype=torch.float32)
