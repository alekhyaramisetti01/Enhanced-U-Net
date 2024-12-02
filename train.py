import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from M2ANET_Arch import MultiAttentionUNet
from data_loader import MedicalImageDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # You can adjust the learning rate if needed

# Dataset and DataLoader for training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

image_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\train_image"
mask_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\train_mask"
dataset = MedicalImageDataset(image_dir, mask_dir, transform=transform)
print(f"Dataset size: {len(dataset)}")  # Check dataset size

# Set batch size to 8
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}]: Loss = {loss.item()}")

    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss / len(dataloader)}")

    # Save the model after each epoch
    model_save_path = r"C:\Users\ramis\Desktop\PROJ\Trained_models\M2ANET_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved after epoch {epoch+1}")
