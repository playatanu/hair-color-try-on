import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from dataset import HairDataset
from unet import UNet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = HairDataset("dataset/train/photos", "dataset/train/masks", transform=transform)
val_dataset = HairDataset("dataset/val/photos", "dataset/val/masks", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

torch.manual_seed = 42
np.random.seed = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet().to(device)
model.load_state_dict(torch.load("models/weights/hair_model.pth", weights_only=True))
model.eval()

criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 20

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
    torch.save(model.state_dict(), "hair_model.pth")

print("Done!")

if __name__ == "__main__":
    print(len(train_dataset), train_dataset[0][0].shape )
