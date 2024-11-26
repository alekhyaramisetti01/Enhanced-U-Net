import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from M2ANET_Arch import MultiAttentionUNet
from data_loader import MedicalImageDataset
from loss_functions import DiceLoss
from torchvision import transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)

# Loss and optimizer
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
train_dataset = MedicalImageDataset("path_to_train_images", "path_to_train_masks", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_dataloader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_dataloader)}")
