import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1️⃣ Define a simple ConvNet
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

model = SimpleCNN()

# 2️⃣ Dummy input image (1 channel, 28x28 like MNIST)
x = torch.randn(1, 1, 28, 28)

# 3️⃣ Forward pass up to conv1
with torch.no_grad():
    conv1_output = model.conv1(x)

# 4️⃣ conv1_output shape: (batch_size, num_filters, height, width)
print(f"Conv1 output shape: {conv1_output.shape}")

# 5️⃣ Plot the feature maps
# Remove batch dimension
feature_maps = conv1_output.squeeze(0)  # shape: (4, 28, 28)

# Plot each filter's output
num_filters = feature_maps.shape[0]
fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))

for i in range(num_filters):
    axes[i].imshow(feature_maps[i].numpy(), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Filter {i+1}')

plt.suptitle('Conv1 Feature Maps')
plt.show()
