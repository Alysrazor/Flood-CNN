import copy
import torch
import numpy as np


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        verbose: bool = False,
        delta: float = 0,
        path: str = "checkpoint.pth",
        restore_best_model: bool = False,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.restore_best_model = restore_best_model

        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model_state: dict | None = None

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1

            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

                if self.restore_best_model and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print("[EARLY STOPPING] Restoring best model weights.")
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased "
                  f"({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")

        torch.save(model.state_dict(), self.path)

        if self.restore_best_model:
            self.best_model_state = copy.deepcopy(model.state_dict())

        self.val_loss_min = min(val_loss, self.val_loss_min)
