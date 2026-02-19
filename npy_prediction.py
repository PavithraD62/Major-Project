import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.base_layers = list(base_model.children())

        # Encoder layers (ResNet backbone)
        self.enc1 = nn.Sequential(*self.base_layers[:3])
        self.enc2 = nn.Sequential(*self.base_layers[3:5])
        self.enc3 = self.base_layers[5]
        self.enc4 = self.base_layers[6]
        self.enc5 = self.base_layers[7]

        # Decoder layers
        self.dec4 = self._decoder_block(2048, 1024)
        self.dec3 = self._decoder_block(1024, 512)
        self.dec2 = self._decoder_block(512, 256)
        self.dec1 = self._decoder_block(256, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # Final output layer

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder forward pass with skip connections
        dec4 = self.dec4(enc5) + enc4
        dec3 = self.dec3(dec4) + enc3
        dec2 = self.dec2(dec3) + enc2
        dec1 = self.dec1(dec2) + enc1
        output = self.upsample(dec1)
        output = self.final(output)

        return output

# 5. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetUNet().to(device)

def predict_and_save_height_maps_npy(model, input_dir, output_dir, device):
    """
    Predict height maps for all images in a directory and save them as .npy files.
    """
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read as RGB
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Convert to tensor and move to device
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Predict the height map
        with torch.no_grad():
            predicted_hmap = model(img_tensor).cpu().numpy()[0, 0]  # Convert to numpy

        # Create label matrix (256x256x10), where only the first class is set to 1
        label_map = np.zeros((256, 256, 10), dtype=np.float32)
        label_map[:, :, 0] = 1  # First terrain class set to 1

        # Structure the data for saving
        data = {
            "input": img,  # Original image (256, 256, 3)
            "label": label_map,  # (256, 256, 10), only first class set to 1
            "height": predicted_hmap * 10,  # Predicted height map (scaled)
        }

        # Save as .npy file with the same name as the input image
        npy_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".npy")
        np.save(npy_path, data)

        print(f"✅ Saved: {npy_path}")

# Define paths
input_dir = "C:/Planetary_Terrains"
output_dir = "C:/Planetary_Terrains/Planetary_Terrains/predicted_npy"

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("C:/Planetary_Terrains/Planetary_Terrains/resnet_mode.pth", map_location=device))
model.to(device)

# Predict and save height maps as NPY files
predict_and_save_height_maps_npy(model, input_dir, output_dir, device)

