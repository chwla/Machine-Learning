import torch
import torch.nn as nn

# Define GELU
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# Define example deep NN with optional shortcut
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # residual connection
            else:
                x = layer_output
        return x

# Gradient inspection helper
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])  # shape match
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    # zero gradients each time
    model.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} gradient mean: {param.grad.abs().mean().item():.6f}")
    print("-" * 40)

# Run both versions
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print("Without shortcut:")
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print("With shortcut:")
print_gradients(model_with_shortcut, sample_input)
