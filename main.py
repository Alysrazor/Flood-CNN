import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from callback import EarlyStopping
from data import train_loader, test_loader, test_ds
from model import save_model, FloodNet10
from train import train, validate, model_pred
from visualization import save_plots

MODEL_DIR = Path(__file__).resolve().parent / "models"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    net: nn.Module,
    early: EarlyStopping,
    dev: torch.device,
    filename: str = 'model.pth',
):
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=1e-3, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5,
        threshold=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    epochs: int = 50

    for epoch in range(epochs):
        print(f"\n[INFO]: Epoch {epoch + 1} of {epochs}")
        train_ep_loss, train_ep_acc = train(
            net, train_loader, optimizer, criterion, dev
        )
        valid_ep_loss, valid_ep_acc = validate(
            net, test_loader, criterion, dev)

        scheduler.step(valid_ep_loss)

        early(valid_ep_loss, model)

        train_loss.append(train_ep_loss)
        valid_loss.append(valid_ep_loss)
        train_acc.append(train_ep_acc)
        valid_acc.append(valid_ep_acc)

        print(f"\nTraining loss: {train_ep_loss:.4f} | "
              f"Training Accuracy: {train_ep_acc:.2f}%")
        print(f"\nValidation loss: {valid_ep_loss:.4f} | "
              f"Validation Accuracy: {valid_ep_acc:.2f}%")
        print("-" * 50)

        if early.early_stop:
            print("--- Early Stopping. End of training ---")
            break
        time.sleep(1)

    save_model(net, MODEL_DIR / filename)
    model_pred(net, test_ds, train_loader, device)
    save_plots(train_acc, valid_acc, train_loss, valid_loss)


if __name__ == "__main__":
    set_seed()
    torch.set_float32_matmul_precision('high')

    # early_stopping = EarlyStopping(patience=5, verbose=True, path=str(MODEL_DIR / "best_floodnet.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FloodNet(num_classes=3).to(device)

    # train_model(model, early_stopping, device, "floodnet.pth")

    # model = FloodNet5(num_classes=3).to(device)
    # early_stopping = EarlyStopping(patience=5, verbose=True, path=str(MODEL_DIR / "best_floodnet5.pth"))
    # train_model(model, early_stopping, device, "floodnet5.pth")

    model = FloodNet10(num_classes=3).to(device)
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        path=str(MODEL_DIR / "best_floodnet10.pth"),
        restore_best_model=True
    )
    train_model(model, early_stopping, device, "floodnet10.pth")
