import os
from PIL import Image
from torch.utils.data import Dataset

class HairDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.img_dir + '/' + self.images[idx]
        mask_path = self.mask_dir + '/' + self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
    

if __name__ == "__main__":
    pass