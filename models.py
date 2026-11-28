import torch

# Set random seed for repoducibility
torch.manual_seed(42)

# Sample input
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

# Define Dimensions
input_dim = 1
hidden_dim = 4
output_dim = 1

# Initialize parameters(weights and bias)
w1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
w2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
b1 = torch.randn(hidden_dim, requires_grad=True)
b2 = torch.randn(output_dim, requires_grad=True)

# Defining the prediction function(no activation)
def predict_linear(x):
    hidden = x @ w1 + b1
    output = hidden @ w2 + b2
    return output

def my_relu(x):
     return torch.clamp(x, min=0)

# Defining the prediction function with activation functions
def predict_activated(x):
    """
    In order to help models better learn the nonlinear relationships in data, we
    use activation function to achieve nonlinearity.
    Activation functions are only pass to hidden layers and never to the output layer
    """
    hidden = my_relu((x @ w1) + b1)
    output = (hidden @ w2) + b2
    return output

y_pred_lin = predict_linear(x)
y_pred_act = predict_activated(x)

print("Predictions for Y with a linear model: " ,y_pred_lin, "\n" \
      "Predictions for Y with a Non-linear model: ", y_pred_act)

def mse_loss(prediction, targets):
    return ((prediction, targets) ** 2).mean()