import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from M2ANET_Arch import MultiAttentionUNet
from data_loader import MedicalImageDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

image_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\test-image"
mask_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\test-mask"
dataset = MedicalImageDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader)}")
