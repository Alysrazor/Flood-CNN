import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

from model import FloodNet, FloodNet10


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)

    for inputs, targets in pbar:
        inputs, targets = (inputs.to(device, non_blocking=True),
                           targets.to(device, non_blocking=True))

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()
    print("\nValidating models...")

    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = (inputs.to(device, non_blocking=True),
                               targets.to(device, non_blocking=True))

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        return total_loss / total, 100.0 * correct / total

def model_pred(
    model: nn.Module,
    dataset: ImageFolder,
    loader: DataLoader,
    device: torch.device
) -> None:
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = (inputs.to(device, non_blocking=True),
                               targets.to(device, non_blocking=True))

            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    class_names = dataset.classes

    print("\n[INFO]: Classification Report")
    if len(y_true) == len(y_pred):
        print(classification_report(y_true, y_pred, target_names=class_names))

        print("\n[INFO]: Confusion Matrix")
        print(confusion_matrix(y_true, y_pred))
    else:
        print(f"[ERROR]: Dimensiones no coinciden. True: {y_true}, Predicted: {y_pred}")