import cv2
import torch
import numpy as np
import os
import supervision as sv
from collections import Counter

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Set the device for CUDA if available
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def binarize_mask(image):
    """Convert annotated image to binary mask."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    pixels_tuple = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixels_tuple)
    most_common_color = color_counts.most_common(1)[0][0]

    # Create masks based on most common color.
    mask_most_common = np.all(image_rgb == most_common_color, axis=-1)
    mask_other = ~mask_most_common

    # Replace colors to create binary masks.
    image_rgb[mask_most_common] = [0] * 3  # Black for most common color.
    image_rgb[mask_other] = [255] * 3       # White for others.

    return cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

def create_mask(image_path):
    """Generate a mask for a given image."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"Image not found at {image_path}")
        return None
    
    # Load the model checkpoint from the specified path
    sam_model = sam_model_registry["vit_h"](checkpoint="D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\sam_ckpt\sam_vit_h_4b8939 (1).pth").to(device=DEVICE) 
    
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    
    masks = mask_generator.generate(image)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections.from_sam(sam_result=masks)

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    return binarize_mask(annotated_image)

def process_dataset(input_folder, output_folder):
    """Process all images in the input folder and save masks to output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)

                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                mask = create_mask(input_path)
                if mask is not None:
                    cv2.imwrite(output_path, mask)
                    print(f"Saved edge mask for {input_path} to {output_path}")

# Example usage 
input_folder = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\dataset'  # Input folder path with dataset images 
output_folder = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\masks'  # Output folder path 
process_dataset(input_folder, output_folder)