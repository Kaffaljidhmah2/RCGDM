import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
from dataset import build_dataset

# Load the pretrained ResNet-18 model
model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)


# Replace the final layer with a linear layer of scalar output
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
torch.save(model, "reward_model.pth")


my_dataset = build_dataset(do_transform=True)

# DataLoader for the dataset
data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, shuffle=False, num_workers=8)


def add_gaussian_noise(output, mean=0, std=0.1):
    noise = np.random.normal(mean, std, output.shape)
    return output + noise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.eval()

outputs_with_noise = []

with torch.no_grad():
    for data, _ in data_loader:
        data = data.to(device)

        # Compute the scalar output for each instance
        output = model(data).cpu().numpy()

        # Add Gaussian noise to the output
        noisy_output = add_gaussian_noise(output)

        # Save the result
        outputs_with_noise.append(noisy_output)

outputs_with_noise = np.concatenate(outputs_with_noise, axis=0)
print(outputs_with_noise)
np.save('cifar10_outputs_with_noise.npy', outputs_with_noise)

