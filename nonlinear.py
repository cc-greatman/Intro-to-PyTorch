import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Input tensor (4 samples, 1 feature)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

# Define dimensions
input_dim = 1
hidden_dim = 3
output_dim = 1

# Initialize parameters
w1 = torch.randn(input_dim, hidden_dim,requires_grad=True) # Here w has a shape of (1, 3)
w2 = torch.randn(hidden_dim, output_dim, requires_grad=True) # Here w has a shape of (3, 1)
b1 = torch.randn(hidden_dim, requires_grad=True) # Here w has a shape of (3,)
b2 = torch.randn(output_dim, requires_grad=True) # Here w has a shape of (1,)

def predict(x):
    hidden = x @ w1 + b1 # @performs matrix multiplication(dot product)
    output = hidden @ w2 + b2
    return output

y_pred = predict(x)
print("Predictions:", y_pred)