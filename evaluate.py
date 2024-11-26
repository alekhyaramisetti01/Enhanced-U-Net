import torch
import numpy as np
from M2ANET_Arch import MultiAttentionUNet
from data_loader import MedicalImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from loss_functions import DiceLoss

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("path_to_saved_model.pth"))  # Load the trained model
model.eval()

# Evaluation metric: Dice Score
def dice_score(pred, target):
    smooth = 1e-5
    pred = (pred > 0.5).float()  # Apply threshold
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + smooth) / (union + smooth)

# DataLoader for validation or test set
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
val_dataset = MedicalImageDataset("path_to_val_images", "path_to_val_masks", transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Evaluate
dice_scores = []
with torch.no_grad():
    for images, masks in val_dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = torch.sigmoid(model(images))  # Get probabilities
        score = dice_score(outputs, masks)
        dice_scores.append(score.item())

print(f"Average Dice Score: {np.mean(dice_scores)}")
