import matplotlib.pyplot as plt
from dataloader import load_data

def visualize_data():
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size=8)

    # Visualize a few examples from the training set
    for images, masks in train_loader:
        for i in range(3):  # Display 3 images and masks
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))  # Convert tensor to HWC format for images
            plt.title("Image")
            plt.axis('off')

            # Squeeze the mask to remove the channel dimension
            plt.subplot(2, 3, i + 4)
            plt.imshow(masks[i].squeeze(), cmap='gray')  # Display mask in grayscale
            plt.title("Mask")
            plt.axis('off')

        plt.show()
        break  # Remove this if you want to visualize more batches

if __name__ == "__main__":
    visualize_data()