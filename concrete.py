import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

input_dim, hidden_dim, output_dim = 8, 16, 1

# Load the dataset
filepath_ = './Dataset/Concrete_Data.csv'
dataframe_ = pd.read_csv(filepath_)

# Separate inputs and targets
inputs = dataframe_.iloc[:, :-1].values
targets = dataframe_.iloc[:, -1].values.reshape(-1,1)

# Standardizing the input
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)

# Converting the inputs to tensor
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
output_tensor = torch.tensor(targets, dtype=torch.float32)

# Set seed and shuffle
torch.manual_seed(32)
num_samples = inputs_tensor.shape[0]
indices = torch.randperm(num_samples)

# Spitting the dataset (getting ids)
split_idx = int(0.8 * num_samples)
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

# Create splits
inputs_train = inputs_tensor[train_idx]
outputs_train = output_tensor[train_idx]
inputs_test = inputs_tensor[test_idx]
outputs_test = output_tensor[test_idx]

class ConcreteMLP(nn.Module):
    def __init__(self, input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim):
        super(ConcreteMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initializing model to a variable
model = ConcreteMLP()

# Defining the Loss function
mse_loss = nn.MSELoss()

# Defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Number of training epochs
epochs = 300

# List to store traing losses
losses_train = []

# Start the training loop
for epoch in range(epochs):

    # Forward Pass
    predictions = model(inputs_train)

    # Loss Computation
    train_loss = mse_loss(predictions, outputs_train)

    # Backward Pass
    train_loss.backward()

    # Optimization
    optimizer.step()

    # Zero Gradients
    optimizer.zero_grad()

    # Record Loss
    losses_train.append(train_loss.item())

    if (epoch+1) % 30 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]: Loss = {train_loss.item():.4f}")