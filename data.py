from pathlib import Path

import torch
import torchvision.tv_tensors
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

TRAIN_IMAGE_DIR = Path(__file__).resolve().parent / "data" / "train"
TEST_IMAGE_DIR = Path(__file__).resolve().parent / "data" / "test"

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

top, left = 27, 27
height, width = 224 - 27, 224 - 27


def crop_image(img: torchvision.tv_tensors.Image):
    return v2.functional.crop(img, top, left, height, width)


train_transform = v2.Compose([
    v2.ToImage(),
    crop_image,
    v2.RandomResizedCrop(size=(IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

test_transform = v2.Compose([
    v2.ToImage(),
    crop_image,
    v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

train_ds = ImageFolder(root=TRAIN_IMAGE_DIR, transform=train_transform)
test_ds = ImageFolder(root=TEST_IMAGE_DIR, transform=test_transform)

train_loader = DataLoader(
    dataset=train_ds, batch_size=BATCH_SIZE,
    shuffle=True, pin_memory=True,
    num_workers=2, persistent_workers=True)
test_loader = DataLoader(
    dataset=test_ds, batch_size=BATCH_SIZE,
    shuffle=False, pin_memory=True,
    num_workers=2, persistent_workers=True
)
