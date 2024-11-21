import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OHRCDataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []

        # Check if directories exist
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")
        
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        # Load image and mask paths
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(class_dir, img_name)
                        mask_path = os.path.join(mask_class_dir, img_name)

                        # Check if the corresponding mask exists
                        if not os.path.exists(mask_path):
                            print(f"Warning: Mask does not exist for image {img_path}. Skipping.")
                            continue

                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Define transformations for images and masks
transform = transforms.Compose([
    transforms.Resize((1200, 1200)),  # Resize the images to a fixed size
    transforms.ToTensor()               # Convert the image to a tensor
])

mask_transform = transforms.Compose([
    transforms.Resize((1200, 1200)),   # Resize masks to match images
    transforms.ToTensor()                # Convert masks to tensor format (optional)
])

def load_data(train_dir='D:/Engg Stuff/Sem 5/EDAI/EDAI-Project/EDAI-Project/Project/dataset/train', 
               val_dir='D:/Engg Stuff/Sem 5/EDAI/EDAI-Project/EDAI-Project/Project/dataset/val', 
               test_dir='D:/Engg Stuff/Sem 5/EDAI/EDAI-Project/EDAI-Project/Project/dataset/test', 
               batch_size=8,
               num_workers=4):  # Set number of workers for data loading
    
    # Create datasets
    train_data = OHRCDataset(
        root_dir=os.path.join(train_dir, 'images'), 
        mask_dir=os.path.join(train_dir, 'masks'), 
        transform=transform,
        mask_transform=mask_transform  # Pass mask transformations if needed
    )
    
    val_data = OHRCDataset(
        root_dir=os.path.join(val_dir, 'images'), 
        mask_dir=os.path.join(val_dir, 'masks'), 
        transform=transform,
        mask_transform=mask_transform
    )
    
    test_data = OHRCDataset(
        root_dir=os.path.join(test_dir, 'images'), 
        mask_dir=os.path.join(test_dir, 'masks'), 
        transform=transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Example of how to call load_data (you can adjust this part as needed)
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data(batch_size=8)