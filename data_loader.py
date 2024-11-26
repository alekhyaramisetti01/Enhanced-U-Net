import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing input images.
            mask_dir (str): Path to the directory containing ground truth masks.
            transform (callable, optional): Optional transforms to apply to the images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace('.jpg', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = MedicalImageDataset("path_to_images", "path_to_masks", transform=transform)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for images, masks in dataloader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        break
