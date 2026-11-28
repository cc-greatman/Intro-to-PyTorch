import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Generate a dataset 100 point from 0 to 2
x = torch.linspace(0, 2, 100).unsqueeze(1)

# Generating non-linear targets
y = x**2 + 0.1 * torch.rand_like(x)

# Set learning rate and epochs
lr = 0.01
epochs = 300

# Defining dimensions
input_dim = 1
hidden_dim = 4
output_dim = 1

# Store training losses
losses_linear = []
losses_relu = []

# Initialize parameters
w1_lin = torch.randn(input_dim, hidden_dim, requires_grad=True)
w2_lin = torch.randn(hidden_dim, output_dim, requires_grad=True)
b1_lin = torch.randn(hidden_dim, requires_grad=True)
b2_lin = torch.randn(output_dim, requires_grad=True)

w1_relu = torch.randn(input_dim, hidden_dim, requires_grad=True)
w2_relu = torch.randn(hidden_dim, output_dim, requires_grad=True)
b1_relu = torch.randn(hidden_dim, requires_grad=True)
b2_relu = torch.randn(output_dim, requires_grad=True)
def my_relu(x):
    return torch.clamp(x, min=0)

def predict_linear(x):
    hidden = (x @ w1_lin) + b1_lin
    output = (hidden @ w2_lin) + b2_lin
    return output

def predict_relu(x):
    hidden = my_relu((x @ w1_relu) + b1_relu)
    output = (hidden @ w2_relu) + b2_relu
    return output

def mse_loss(preds, labels):
    return ((preds - labels) ** 2).mean()

# Begin training
for epoch in range(epochs):

    # Train the Linear regression model
    lin_pred = predict_linear(x)
    lin_loss = mse_loss(lin_pred, y)

    # Backward pass
    lin_loss.backward()

    # Updating parameters
    with torch.no_grad():
        w1_lin -= lr * w1_lin.grad
        w2_lin -= lr * w2_lin.grad
        b1_lin -= lr * b1_lin.grad
        b2_lin -= lr * b2_lin.grad

        # Gradient reset
        w1_lin.grad.zero_()
        w2_lin.grad.zero_()
        b1_lin.grad.zero_()
        b2_lin.grad.zero_()

    losses_linear.append(lin_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {lin_loss:.4f}")

for epoch in range(epochs):

    # Training the Non-linear model
    nonlin_pred = predict_relu(x)
    relu_loss = mse_loss(nonlin_pred, y)

    # Backward Pass
    relu_loss.backward()

    # Update the parameter
    with torch.no_grad():
        w1_relu -= lr * w1_relu.grad
        w2_relu -= lr * w2_relu.grad
        b1_relu -= lr * b1_relu.grad
        b2_relu -= lr * b2_relu.grad
       
        # Gradient reset
        w1_relu.grad.zero_()
        w2_relu.grad.zero_()
        b1_relu.grad.zero_()
        b2_relu.grad.zero_()

    losses_relu.append(relu_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {relu_loss:.4f}")

plt.plot(losses_linear, label='Linear -> Linear')
plt.plot(losses_relu, label='Linear -> ReLU -> Linear')
plt.xlabel("Number of Epochs")
plt.ylabel("MSE Loss")
plt.title("Training loss comparison")
plt.legend()
plt.grid(True)
plt.show()
