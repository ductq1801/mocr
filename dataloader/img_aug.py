import albumentations as A
from PIL import Image
import numpy as np

class ImgAugTransform:
    def __init__(self):
        self.aug = A.Compose([
            A.Rotate(limit=9,p=0.2),
            A.ColorJitter(p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Perspective(scale=(0.01, 0.05))
            ])

    def __call__(self, img):
        img = np.array(img)
        transformed = self.aug(image=img)
        img = transformed["image"]
        img = Image.fromarray(img)
        return img