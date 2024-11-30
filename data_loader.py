import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing input images.
            mask_dir (str): Path to the directory containing ground truth masks.
            transform (callable, optional): Optional transform to apply to the images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def get_valid_mask_path(self, image_filename):
        """
        Try different extensions to find the corresponding mask file.
        Args:
            image_filename (str): The image file name.
        Returns:
            str: Path to the valid mask file, or None if not found.
        """
        extensions = ['.jpg', '.jpeg', '.png']  # Check common image file extensions
        for ext in extensions:
            mask_filename = image_filename.replace('.jpg', ext)  # Replace .jpg with mask extension
            mask_path = os.path.join(self.mask_dir, mask_filename)
            if os.path.exists(mask_path):  # If mask exists, return path
                return mask_path
        return None  # Return None if no matching mask is found

    def __getitem__(self, idx):
        # Get the image file path
        img_path = os.path.join(self.image_dir, self.image_list[idx])

        # Try to get the corresponding mask for the image
        mask_path = self.get_valid_mask_path(self.image_list[idx])
        
        if mask_path is None:
            print(f"Mask file not found for image: {img_path}")
            mask = Image.new("L", Image.open(img_path).size)  # Create an empty mask (all zeros) if not found
        else:
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Open the image file
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images and masks to smaller size
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Define the directory paths (update these to your actual dataset paths)
    image_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\train_image"
    mask_dir = r"C:\Users\ramis\Desktop\Thyroid Dataset\tn3k\train_mask"
    dataset = MedicalImageDataset(image_dir, mask_dir, transform=transform)

    # Load a batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, masks in dataloader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        break  # Just printing the first batch for testing
