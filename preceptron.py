import torch
import torch.nn as nn

# Initiating Deminsions
input_dim = 8
hidden_dim = 16
output_dim = 1

# Designing a simple Multi-layer Preceptron
simple_mlp = nn.Sequential(
    # Linear layer mapping from a-dim input to b-dim output
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

# Simple MLP with one hidden layer
# The class inherits from nn.Module
class ConcreteMLP(nn.Module):
    def __init__(self, input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim):
        super(ConcreteMLP, self).__init__()

        # Define the layers in the preceptron
        self.fc1 = nn.Linear(input_size, hidden_size) # First hidden layer
        self.relu = nn.ReLU() # Activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        x = self.fc1(x) # Applying the input layer
        x = self.relu(x) # Applying the activation layer
        x = self.fc2(x) # Applying the output layer
        return x

# Create model instance
model = ConcreteMLP()
print(model)

# Extending the model: Two hidden layers
class DeeperConcreteMLP(nn.Module):
    def __init__(self):
        super(DeeperConcreteMLP, self).__init__()

        self.fc1 = nn.Linear(8, 32) # First hidden layer
        self.relu1 = nn.ReLU() 
        self.fc2 = nn.Linear(32, 16) # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1) # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

deep_model = DeeperConcreteMLP()
print(deep_model)