import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from dataset import HairDataset
from model import UNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train(num_epochs=20, pre_weights=None):

    torch.manual_seed = 42
    np.random.seed = 42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data transformation
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])

    # dataloader
    train_dataset = HairDataset("dataset/train/photos", "dataset/train/masks", transform=transform)
    val_dataset = HairDataset("dataset/val/photos", "dataset/val/masks", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # model initialization
    model = UNet().to(device)

    if pre_weights != None:
        model.load_state_dict(torch.load(pre_weights, weights_only=True, map_location=device))
        
    model.eval()

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True) 

        save_path = os.path.join(save_dir, "hair_seg_model.pth")
        torch.save(model.state_dict(), save_path)

    print("Done!")

if __name__ == "__main__":
    train(num_epochs=20, pre_weights="hair_segmentation/weights/hair_seg_model.pth")
