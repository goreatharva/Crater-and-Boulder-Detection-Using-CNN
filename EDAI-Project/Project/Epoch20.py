import matplotlib.pyplot as plt
from PIL import Image

def visualize_images(original_image_paths, mask_image_paths):
    # Create a figure to display images
    num_images = len(original_image_paths)
    plt.figure(figsize=(15, 5 * num_images))  # Adjust figure size based on number of images

    for i in range(num_images):
        # Load original image
        original_image = Image.open(original_image_paths[i]).convert("RGB")
        # Load mask image
        mask_image = Image.open(mask_image_paths[i]).convert("L")  # Convert to grayscale if it's a mask

        # Display original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(original_image)
        plt.title(f"Original Image {i + 1}")
        plt.axis('off')

        # Display masked image
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(mask_image, cmap='gray')  # Use gray colormap for the mask
        plt.title(f"Actual Mask {i + 1}")
        plt.axis('off')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   
    original_image_paths = [
       r'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\images\20190906\ch2_ohr_ncp_20190906T1246532096_b_brw_d18_0.png',
       r'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\dataset\test\images\20200229\ch2_ohr_ncp_20200229T0739312111_b_brw_d18_2.png',
       r'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\images\20190906\ch2_ohr_ncp_20190906T1246532096_b_brw_d18_3.png'
    ]

    mask_image_paths = [
       r'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\masks\20190906\ch2_ohr_ncp_20190906T1246532096_b_brw_d18_0.png',
       r'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\masks\20200229\ch2_ohr_ncp_20200229T0739312111_b_brw_d18_1.png',
       r'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project\data_processed\masks\20190906\ch2_ohr_ncp_20190906T1246532096_b_brw_d18_3.png'
    ]

    # Call the function to visualize images
    visualize_images(original_image_paths, mask_image_paths)