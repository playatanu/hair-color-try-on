from models.unet import UNet

import torch
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def image_to_mask(image):
    model = UNet()
    model.load_state_dict(torch.load("models/weights/hair_model.pth", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        image = transform(image)
        output = model(image.unsqueeze(0)).squeeze(0).squeeze(0)
        output = output.detach().numpy().astype(np.uint8)
        return 1 - output
