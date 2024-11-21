import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def check_image_width(folder_path, expected_width):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    height, width, _ = image.shape
                    if width != expected_width:
                        print(f"File {image_path} has width {width}, expected {expected_width}.")

def collect_image_mask_pairs(image_folder, mask_folder):
    image_files = []
    mask_files = []
    
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png', '.jpg')):  # Include other formats if necessary
                rel_path = os.path.relpath(os.path.join(root, file), start=image_folder)
                mask_path = os.path.join(mask_folder, rel_path)
                if os.path.exists(mask_path):
                    image_files.append(os.path.join(root, file))
                    mask_files.append(mask_path)
                    print(f"Found image: {os.path.join(root, file)}")
                    print(f"Found mask: {mask_path}")

    return image_files, mask_files

def split_data(image_files, mask_files, output_folder, train_ratio=0.8, val_ratio=0.1):
    if len(image_files) == 0:
        raise ValueError("No images found to split.")

    train_image_files, test_image_files = train_test_split(image_files, test_size=(1 - train_ratio))
    val_image_files, test_image_files = train_test_split(test_image_files,
                                                          test_size=(val_ratio / (val_ratio + (1 - train_ratio))))

    def copy_files(files, src_folder, dst_folder):
        for file in files:
            rel_path = os.path.relpath(file, start=src_folder)
            dest_path = os.path.join(dst_folder, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'masks'), exist_ok=True)

    copy_files(train_image_files, image_folder,
               os.path.join(output_folder, 'train', 'images'))
    copy_files([f.replace(image_folder, mask_folder) for f in train_image_files],
               mask_folder,
               os.path.join(output_folder, 'train', 'masks'))

    copy_files(val_image_files,
               image_folder,
               os.path.join(output_folder, 'val', 'images'))
    copy_files([f.replace(image_folder, mask_folder) for f in val_image_files],
               mask_folder,
               os.path.join(output_folder, 'val', 'masks'))

    copy_files(test_image_files,
               image_folder,
               os.path.join(output_folder, 'test', 'images'))
    copy_files([f.replace(image_folder, mask_folder) for f in test_image_files],
               mask_folder,
               os.path.join(output_folder, 'test', 'masks'))

# Configuration
image_folder = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\images'  # Input folder path with processed images
mask_folder = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\masks'  # Input folder path with masks
output_folder = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\dataset'  # Output folder path where datasets are stored
expected_width = 1200

# Check image widths
check_image_width(image_folder, expected_width)
check_image_width(mask_folder, expected_width)

# Collect image and mask pairs
image_files, mask_files = collect_image_mask_pairs(image_folder, mask_folder)

# Print counts before splitting
print(f"Number of images collected: {len(image_files)}")
print(f"Number of masks collected: {len(mask_files)}")

# Split the dataset
split_data(image_files, mask_files, output_folder)