import torch
import matplotlib.pyplot as plt
from M2ANET_Arch import MultiAttentionUNet
from data_loader import MedicalImageDataset
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("path_to_saved_model.pth"))
model.eval()

# DataLoader for images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = MedicalImageDataset("path_to_images", "path_to_masks", transform=transform)

# Visualize Predictions
def visualize_prediction(image, mask, pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0))  # Convert CHW to HWC
    axes[0].set_title("Input Image")
    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred.squeeze(), cmap="gray")
    axes[2].set_title("Predicted Mask")
    plt.show()

# Predict and visualize
for i in range(5):  # Show 5 predictions
    image, mask = dataset[i]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        pred = torch.sigmoid(model(image))
    visualize_prediction(image.squeeze().cpu(), mask, pred.cpu())
