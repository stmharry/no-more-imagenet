from dataclasses import dataclass, field

import torch
from torchvision import transforms

from app.schemas.transforms import Transform


@dataclass
class GaussianNoise:
    sigma: float
    to_tensor: Transform = field(default_factory=transforms.ToTensor)
    to_pil_image: Transform = field(default_factory=transforms.ToPILImage)

    def __call__(self, img):
        img = self.to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            img = img + self.sigma * torch.rand_like(img)
            img = img.squeeze()

        img = self.to_pil_image(img)
        return img
