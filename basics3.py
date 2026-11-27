import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

filepath_ = './Dataset/Concrete_Data.csv'
data = pd.read_csv(filepath_)

def mse_loss(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def mae_loss(predictions, targets):
    return torch.mean((predictions - targets))

# Seperate inputs and targets
# Get all values except the last column and reshape to numpy dataframe
inputs = data.iloc[:, :-1].values
# Get the last column and reshape the values to numpy dataframe
targets = data.iloc[:, -1].values.reshape(-1, 1)

# 3. Normalize input features
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)

# 4. Convert to PyTorch tensors
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# 5. Shuffle and split into training(80%) and test(20%) sets
torch.manual_seed(42) # For repoducibilty
n_samples = inputs_tensor.shape[0]
indices = torch.randperm(n_samples)
split_idx = int(n_samples * 0.8) # Setting the standard for splitting

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

x_train = inputs_tensor[train_indices]
x_test = inputs_tensor[test_indices]
y_train = targets_tensor[train_indices]
y_test = targets_tensor[test_indices]

# 6. Initialize model parameters
num_features = x_train.shape[1]
# We tell PyTorch which tensors to track for gradients
w = torch.randn((num_features, 1), requires_grad=True)
w.data *= 0.01 # scale down initial values
b = torch.zeros((1,), requires_grad=True)

print(f"Training set size: {x_train.shape[0]}\n" \
      f"Test set size: {x_test.shape[0]}\n " \
      f"Weight shape: {w.shape}\n" \
      f"Bias shape: {b.shape}")

def predict(x):
    return b + torch.matmul(x, w)

# Set learning rate and number of training epochs
learning_rate = 0.001
epochs = 500

losses = []

# Training loop
for epoch in range(epochs):
    # 1. Forward pass: compute predictions
    y_pred = predict(x_train)

    # 2. Compute loss
    loss = mse_loss(y_pred, y_train)
    losses.append(loss.item()) # Track the loss

    # 3. Backward pass: compute gradients
    loss.backward()

    # 4. Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 5. Zero gradients
    w.grad.zero_()
    b.grad.zero_()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training loss over Epochs")
# plt.grid(True)
# plt.show()

y_new_pred = predict(x_test)

test_mae = mae_loss(y_new_pred, y_test)

print(f"MAE loss: {test_mae.item():.4f}")

