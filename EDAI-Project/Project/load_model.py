import torch
from models.ESAU_net import ESAU  # Ensure this file exists and is implemented correctly
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_model(model_path):
    model = ESAU()  # Instantiate your model architecture
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load weights with weights_only=True
    model.eval()  # Set to evaluation mode
    return model

def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((1200, 1200)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)  # Get predictions

    return output

def visualize_prediction(image_path, output):
    image = Image.open(image_path).convert("RGB")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Squeeze the output to remove singleton dimensions and ensure it's in the right shape
    mask = output.squeeze(0).cpu().numpy()  # Remove batch dimension
    if mask.shape[0] == 1:  # If the output has a single channel
        mask = mask[0]  # Take only the first channel
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')  # Assuming output is a mask
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    model_path = 'model_epoch_1.pth'  # Path to your saved model
    
    # List of test images (update paths accordingly)
    test_images = [
       'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project/dataset/test/images/20200824/ch2_ohr_ncp_20200824T0806596861_b_brw_d18_6.png',
       'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project/dataset/test/images/20230823/ch2_ohr_ncp_20230823T1647285085_b_brw_n18_6.png',
       'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project/dataset/test/images/20230823/ch2_ohr_ncp_20230823T1647285315_b_brw_n18_0.png',
       # Add more test images as needed
    ]

    model = load_model(model_path)

    for test_image in test_images:
        output = predict(model, test_image)
        visualize_prediction(test_image, output)