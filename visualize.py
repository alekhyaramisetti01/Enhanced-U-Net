import torch
import time
import matplotlib.pyplot as plt
from M2ANET_Arch import MultiAttentionUNet
from data_loader import MedicalImageDataset
from torchvision import transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize and load the model
print("Initializing the model...")
model = MultiAttentionUNet(in_channels=3, out_channels=1).to(device)

# Path to the saved model
model_path = r"C:\Users\ramis\Desktop\PROJ\Trained_models\M2ANET_epoch_5.pth"
print(f"Loading model weights from: {model_path}")
try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please check the path and file name.")
    exit()

# DataLoader for images
print("Setting up the dataset and transformations...")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure resizing matches training size
    transforms.ToTensor(),
])

image_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\test_image"
mask_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\test_mask"
print(f"Loading dataset from:\n  Images: {image_dir}\n  Masks: {mask_dir}")
dataset = MedicalImageDataset(image_dir, mask_dir, transform=transform)
print(f"Dataset loaded successfully. Total samples: {len(dataset)}")

# Visualization Function
def visualize_prediction(image, mask, pred):
    print("Visualizing prediction...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if image.shape[0] == 3:  # RGB
        axes[0].imshow(image.permute(1, 2, 0))  # Convert CHW to HWC
    else:  # Grayscale
        axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title("Input Image")
    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred.squeeze(), cmap="gray")
    axes[2].set_title("Predicted Mask")
    plt.show()
    print("Visualization completed.")

# Predict and visualize
print("Starting prediction and visualization process...")
try:
    for i in range(5):  # Show 5 predictions
        print(f"Processing sample {i + 1}/{min(5, len(dataset))}...")

        # Load image and mask
        image, mask = dataset[i]
        print(f"Loaded image and mask for sample {i + 1}.")
        
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        print("Image moved to device for prediction.")
        
        # Measure time taken for prediction
        start_time = time.time()
        with torch.no_grad():
            pred = torch.sigmoid(model(image))  # Get probabilities
        end_time = time.time()
        print(f"Prediction time for sample {i + 1}: {end_time - start_time:.4f} seconds")
        
        visualize_prediction(image.squeeze().cpu(), mask, pred.squeeze().cpu())
        print(f"Visualization for sample {i + 1} completed.\n")

except IndexError:
    print("Error: Dataset index out of range. Ensure you have at least 5 samples in the dataset.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("Prediction and visualization process completed.")
