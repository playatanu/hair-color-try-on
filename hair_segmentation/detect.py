import torch
import numpy as np
from .model import UNet
from torchvision import transforms

def image_to_mask(image, model_path):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = UNet()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
    
    with torch.no_grad():
        image = transform(image)
        output = model(image.unsqueeze(0)).squeeze(0).squeeze(0)
        output = output.detach().numpy().astype(np.uint8)
        return 1 - output

if __name__ == "__main__":
    pass