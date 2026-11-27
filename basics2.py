import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

# Step 1: Tiny toy dataset
# Inputs
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) 
# Ground truths
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Step 2: Manually initialized weights and bias
w = torch.tensor([[0.5]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# Step 3: Define the predict function(y = wx +b)
def predict(x):
    return b + torch.matmul(x,w)

# Step 4: Make predictions
y_pred = predict(x)

# Step 5: Show results
print(f"Predictions: {y_pred}")

def mse_loss(pred, val):
    return torch.mean((pred - val) ** 2)

loss = mse_loss(y_pred, y)
print(f"MSE loss: {loss.item()}")
