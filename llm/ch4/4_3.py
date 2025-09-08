import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define GELU activation
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=x.dtype)) *
            (x + 0.044715 * x**3)
        ))

def plot_activations():
    # Initialize activations
    gelu = GELU()
    relu = nn.ReLU()

    # Input range
    x = torch.linspace(-3, 3, 100)

    # Compute activations
    y_gelu = gelu(x)
    y_relu = relu(x)

    # Plot
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_activations()
