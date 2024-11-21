import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.ESAU_net import ESAU  # Ensure this file exists and is implemented correctly
from dataloader import load_data  # Ensure this function is implemented correctly

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants for absolute data directories and batch size
TRAIN_DIR = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project/dataset/train'
VAL_DIR = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project/dataset/val'
TEST_DIR = 'D:\Engg Stuff\Sem 5\EDAI\EDAI-Project\EDAI-Project\Project/dataset/test'
BATCH_SIZE = 1  # Reduced batch size

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((1200, 1200)),  # Resize images to a fixed size
    transforms.ToTensor(),              # Convert images to tensor format
])

# Load datasets using custom dataloader
print("Loading data...")
train_loader, val_loader, test_loader = load_data(TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE)

# Model Initialization
print("Initializing model...")
model = ESAU().to(device)  # Move model to GPU if available

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()  # Adjust based on your task (e.g., binary classification)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Mixed Precision
from torch.cuda.amp import GradScaler, autocast

def validate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():  # Disable gradient calculation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item()

    avg_val_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

def train_model_with_logging(model, train_loader, val_loader):
    num_epochs = 3 # Set your number of epochs here
    print("Starting training loop...")
    
    scaler = GradScaler()  # Initialize gradient scaler for mixed precision

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()  # Set the model to training mode
        
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
            
            # Move data and target to GPU if available
            data, target = data.to(device), target.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                output = model(data)
                loss = loss_function(output, target)  # Compute the loss
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            scaler.scale(loss).backward()  # Backpropagation with scaling
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scale for next iteration
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")

        # Validate after each epoch
        validate_model(model, val_loader)

        # Save model checkpoint (optional)
        checkpoint_path = f'model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved: {checkpoint_path}')

        # Clear cache after each epoch (optional)
        torch.cuda.empty_cache()

    print("Training completed.")

if __name__ == '__main__':
    train_model_with_logging(model, train_loader, val_loader)