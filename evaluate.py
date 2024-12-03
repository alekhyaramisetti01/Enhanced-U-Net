import torch
import numpy as np
from M2ANET_Arch import MultiAttentionUNet  # Your model architecture
from data_loader import MedicalImageDataset  # Your dataset loader
from torchvision import transforms
from torch.utils.data import DataLoader

# Set device configuration (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
print("Initializing the model...")
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)

# Path to the saved model
epochs = 5  # Set this based on the epoch you're evaluating
model_path = r"C:\Users\ramis\Desktop\PROJ\Trained_models\M2ANET_epoch_5.pth"  # Replace with your file path

# Load the trained model weights
print(f"Loading model weights from: {model_path}")
try:
    model.load_state_dict(torch.load(model_path))  # Load model weights
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please check the path and file name.")
    exit()

# Define the Dice Score function
def dice_score(pred, target):
    smooth = 1e-5  # Small constant to avoid division by zero
    pred = (pred > 0.5).float()  # Apply threshold to convert probabilities to binary predictions
    intersection = (pred * target).sum()  # Compute the intersection of prediction and ground truth
    union = pred.sum() + target.sum()  # Compute the union of prediction and ground truth
    return (2 * intersection + smooth) / (union + smooth)

# DataLoader for validation or test set
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to the size used during training
    transforms.ToTensor(),
])

# Paths to the validation/test dataset
val_image_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\test_image"
val_mask_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\test_mask"

# Initialize the dataset and DataLoader for validation
print(f"Loading validation dataset from:\n  Images: {val_image_dir}\n  Masks: {val_mask_dir}")
val_dataset = MedicalImageDataset(val_image_dir, val_mask_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"Validation dataset loaded successfully. Total samples: {len(val_dataset)}")

# List to store the Dice scores for each sample
dice_scores = []

print("Starting evaluation...")
# Disable gradient calculation during inference
with torch.no_grad():
    for idx, (images, masks) in enumerate(val_dataloader):
        print(f"Processing sample {idx + 1}/{len(val_dataloader)}...", end="\r")
        images, masks = images.to(device), masks.to(device)

        # Get model predictions (output probabilities)
        outputs = torch.sigmoid(model(images))  # Apply sigmoid to get probabilities

        # Compute the Dice Score for the sample
        score = dice_score(outputs, masks).item()
        dice_scores.append(score)  # Store the score

        # Print individual Dice score
        print(f"Sample {idx + 1}/{len(val_dataloader)}: Dice Score = {score:.4f}")

print("\nEvaluation completed.")

# Compute the average Dice Score across all samples
average_dice_score = np.mean(dice_scores)
print(f"Average Dice Score: {average_dice_score:.4f}")

# Optional: Save results to a file
output_file = r"C:\Users\ramis\Desktop\PROJ\evaluation_results.txt"
try:
    with open(output_file, "w") as f:
        f.write("Sample Index\tDice Score\n")
        for idx, score in enumerate(dice_scores):
            f.write(f"{idx + 1}\t{score:.4f}\n")
        f.write(f"\nAverage Dice Score: {average_dice_score:.4f}\n")
    print(f"Results saved to {output_file}")
except Exception as e:
    print(f"Error saving results: {e}")
