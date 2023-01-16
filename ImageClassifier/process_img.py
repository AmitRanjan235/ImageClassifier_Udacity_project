from PIL import Image
from utils import test_transforms
import numpy as np
import torch

def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))

    np_image = np.array(img)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np.transpose(np_image, (2, 0, 1))

    return torch.from_numpy(np_image)