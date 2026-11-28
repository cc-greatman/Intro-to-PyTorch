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
b1 = torch.randn(hidden_dim, requires_grad=True) # Here b has a shape of (3,)
b2 = torch.randn(output_dim, requires_grad=True) # Here b has a shape of (1,)

def predict(x):
    hidden = x @ w1 + b1 # @performs matrix multiplication(dot product)
    output = hidden @ w2 + b2
    return output

y_pred = predict(x)
print("Predictions:", y_pred)

# Defining ReLU activation function for intuition
def my_relu(x):
    """
    A simple implementation of the Rectified Linear Unit (ReLU) activation function. 
    It returns the input tensor x if its value is greater than 0; otherwise, it returns 0.
    """
    return torch.clamp(x, min=0) 

# Test on a simple tensor
test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input: ", test_x, "\n" \
      "ReLU output: ", my_relu(test_x))

# Defining Sigmoid activation function for intuition too 😂
def my_sigmoid(x):
    """
    A simple implementation of the Sigmoid activation function.
    It compresses values into the range (0, 1)
    """
    return 1 / (1 + torch.exp(-x))

# Testing for Sigmoid activation function
print("Input: ", test_x, "\n" \
      "Sigmoid output: ", my_sigmoid(test_x))

# Defining Tanh activation function for more intuition
def my_tanh(x):
    """
    A simple implementation of the Hyperbolic Tangent activation function.
    it compresses values in the range (-1, 1)
    """
    a, b = torch.exp(x) ,torch.exp(-x)
    return (a - b) / (a + b) 

# Testing for Tanh activation function
print("Input: ", test_x, "\n" \
      "Tanh output: ", my_tanh(test_x))