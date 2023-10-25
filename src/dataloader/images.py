import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

def get_images(folder_path, size):
    mean = [0.5, 0.5, 0.5]
    sd = [0.5, 0.5, 0.5]
    # mean = [0.485, 0.456, 0.406]
    # sd = [0.229, 0.224, 0.225]
    transform_array = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, sd)
    ])

    images = []
    for file_path in os.listdir(folder_path):
        if ".jpg" not in file_path and ".png" not in file_path:
            continue
        img = Image.open(os.path.join(folder_path, file_path))
        images.append(transform_array(img))

    return torch.stack(images)

def get_images_2(folder_path, img_names, size):
    mean = [0.5, 0.5, 0.5]
    sd = [0.5, 0.5, 0.5]
    # mean = [0.485, 0.456, 0.406]
    # sd = [0.229, 0.224, 0.225]
    transform_array = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, sd)
    ])

    images = []
    for filename in img_names:
        file_path = os.path.join(folder_path, filename)
        img = Image.open(file_path)
        img = img.convert("RGB")
        images.append(transform_array(img))

    return torch.stack(images)

def get_dataloader(images, batch_size, folder_path=False):
    if folder_path:
        images = get_images(images)
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    return loader
