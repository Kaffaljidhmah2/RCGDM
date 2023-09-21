import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
from dataset import build_dataset


from model import ThreeLayerConvNet
from vae import prepare_image,  encode, sd_model
import wandb


### Configs #### 

lr = 0.001
num_data = 50000
num_epochs = 100


wandb.init(project="guided_dm", config={
    'lr': lr,
    'num_data':num_data,
    'num_epochs':num_epochs
})

device = 'cuda'


# stable diffusion hyperparameters.
latent_dim = 4  
num_inference_steps = 50  


convnet = ThreeLayerConvNet(latent_dim).to(device)

my_dataset = build_dataset(do_transform = False)

encoded_images = []
with torch.no_grad():
    for data, _ in  my_dataset:
        data = prepare_image(data)
        data = data.to(device)

        encoded_img = encode(data)
        encoded_images.append(encoded_img)

encoded_images = torch.cat(encoded_images, dim = 0).cpu()



# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(convnet.parameters(), lr=lr)

# Prepare the training dataset with the encoded images and noisy outputs
outputs_with_noise = np.load('cifar10_outputs_with_noise.npy')
my_targets = torch.tensor(outputs_with_noise, dtype=torch.float32)


assert num_data <= 50000, 'maximum reached.'


train_dataset = torch.utils.data.TensorDataset(encoded_images[:num_data], my_targets[:num_data])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8)

# Train the model

# noisy input
sd_model.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = sd_model.scheduler.timesteps


for epoch in range(num_epochs):
    convnet.train()
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Add random noise to the latent.
        random_sampled_timesteps = timesteps[torch.randint(low=0, high=len(timesteps), size=(inputs.shape[0],), device = device)]
        random_noise =  torch.randn_like(inputs, device = device)
        inputs = sd_model.scheduler.add_noise(original_samples = inputs, noise = random_noise, timesteps = random_sampled_timesteps)

        # Forward pass
        outputs = convnet(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the weights
        optimizer.step()

        epoch_loss += loss.item()

    # Print epoch loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / (i+1):.4f}')
    wandb.log({"train_loss": epoch_loss/(i+1)})

torch.save(convnet, 'convnet.pth')

