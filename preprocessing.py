"""
preprocessing.py — Shared data preprocessing for the SwAV / MNIST (100 labels) project.

Everyone on the team imports this file so that we all use:
  - the same MNIST loading and normalization,
  - the SAME 100 labeled images (10 per digit, single balanced split, fixed seed).

Usage
-----
    from preprocessing import get_data, set_seed, get_device

    set_seed()
    data = get_data()
    data["X_unlabeled"]   # (60000, 1, 28, 28) float, normalized  -> self-supervised pretraining (no labels)
    data["X_lab"], data["y_lab"]   # (100, 1, 28, 28), (100,)  -> the 100 labeled images
    data["X_test"],  data["y_test"]  # (10000, 1, 28, 28), (10000,) -> final evaluation

Run `python preprocessing.py` to print a summary and the split checksum
(everyone should get the same checksum).
"""

import os
import json
import random
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
N_CLASSES = 10
N_PER_CLASS = 10                      # 10 x 10 = 100 labeled images
DATA_DIR = "./data"
SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"labeled_indices_seed{SEED}.json")

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
NORMALIZE = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))   # for augmentation pipelines


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# LOADING & NORMALIZATION
# =============================================================================
def load_mnist(data_dir=DATA_DIR):
    """Raw MNIST as uint8 tensors: X (N, 28, 28), y (N,)."""
    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, train=False, download=True)
    return train.data, train.targets, test.data, test.targets


def to_unit(x_uint8):
    """uint8 (N, 28, 28) -> float in [0, 1], shape (N, 1, 28, 28)."""
    return (x_uint8.float() / 255.0).unsqueeze(1)


def normalize(x_uint8):
    """uint8 (N, 28, 28) -> normalized float (N, 1, 28, 28)."""
    return (to_unit(x_uint8) - MNIST_MEAN) / MNIST_STD


# =============================================================================
# LABELED SPLIT — 100 IMAGES, 10 PER CLASS
# =============================================================================
def sample_labeled_indices(targets, n_per_class=N_PER_CLASS, seed=SEED):
    """Balanced draw: exactly n_per_class indices per digit.
    Uses np.random.RandomState, whose stream is stable across NumPy versions,
    so every machine gets the same indices."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    idx = []
    for c in range(N_CLASSES):
        cls_idx = np.where(targets == c)[0]
        idx.extend(sorted(rng.choice(cls_idx, size=n_per_class, replace=False).tolist()))
    return np.array(idx, dtype=np.int64)


def split_checksum(indices):
    return hashlib.md5(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:10]


def check_balanced(indices, targets, n_per_class=N_PER_CLASS):
    counts = np.bincount(np.asarray(targets)[indices], minlength=N_CLASSES)
    assert len(indices) == N_CLASSES * n_per_class, f"expected {N_CLASSES*n_per_class} indices, got {len(indices)}"
    assert len(set(indices.tolist())) == len(indices), "duplicate indices"
    assert (counts == n_per_class).all(), f"unbalanced split: {counts}"


def get_labeled_indices(y_train, split_file=SPLIT_FILE, n_per_class=N_PER_CLASS, seed=SEED):
    """Load the split from split_file if it exists, otherwise create and save it.
    Commit the JSON file to GitHub: it is the single source of truth."""
    if os.path.exists(split_file):
        with open(split_file) as f:
            saved = json.load(f)
        indices = np.array(saved["indices"], dtype=np.int64)
    else:
        indices = sample_labeled_indices(y_train, n_per_class, seed)
        with open(split_file, "w") as f:
            json.dump({"seed": seed, "n_per_class": n_per_class,
                       "checksum": split_checksum(indices),
                       "indices": indices.tolist()}, f, indent=1)
        print(f"[preprocessing] Created split file: {split_file}")
    check_balanced(indices, y_train, n_per_class)
    return indices


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def get_data(data_dir=DATA_DIR, split_file=SPLIT_FILE):
    """Returns a dict with everything needed for the project.

    X_unlabeled : all 60,000 train images (normalized), used WITHOUT labels.
                  (The 100 labeled images are included, as is standard in semi-supervised learning.)
    X_lab, y_lab: the 100 labeled images (10 per digit).
    X_test, y_test: the 10,000 test images.
    *_raw       : uint8 versions, for augmentation pipelines / visualization.
    """
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    lab_idx = get_labeled_indices(y_train, split_file)
    lab_idx_t = torch.from_numpy(lab_idx)

    return {
        "X_unlabeled": normalize(X_train),
        "X_lab": normalize(X_train[lab_idx_t]),
        "y_lab": y_train[lab_idx_t].clone(),
        "X_test": normalize(X_test),
        "y_test": y_test.clone(),
        "labeled_idx": lab_idx,
        "X_train_raw": X_train,
        "y_train": y_train,       # only for final analysis (clustering metrics, t-SNE colors) — never for training
        "X_test_raw": X_test,
    }


# =============================================================================
# DATASETS / DATALOADERS
# =============================================================================
class UnlabeledMNIST(Dataset):
    """Train images without labels. `transform` receives a float tensor in [0, 1]
    of shape (1, 28, 28) and may return a single tensor or a list of views (multi-crop).
    Add NORMALIZE at the end of your transform pipeline."""

    def __init__(self, X_raw, transform=None):
        self.X = to_unit(X_raw)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        return self.transform(x) if self.transform is not None else (x - MNIST_MEAN) / MNIST_STD


def get_loaders(data, batch_size_lab=100, batch_size_test=1000, num_workers=0):
    """Labeled (100 images) and test loaders on normalized tensors."""
    lab_loader = DataLoader(TensorDataset(data["X_lab"], data["y_lab"]),
                            batch_size=batch_size_lab, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(data["X_test"], data["y_test"]),
                             batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
    return lab_loader, test_loader


def get_unlabeled_loader(data, transform=None, batch_size=256, num_workers=2, drop_last=True):
    """Loader over the 60,000 train images without labels (for SwAV pretraining)."""
    ds = UnlabeledMNIST(data["X_train_raw"], transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      drop_last=drop_last, pin_memory=torch.cuda.is_available())


# =============================================================================
# SUMMARY
# =============================================================================
if __name__ == "__main__":
    set_seed()
    data = get_data()
    print("Unlabeled train :", tuple(data["X_unlabeled"].shape))
    print("Labeled         :", tuple(data["X_lab"].shape), "| per class:",
          torch.bincount(data["y_lab"], minlength=N_CLASSES).tolist())
    print("Test            :", tuple(data["X_test"].shape))
    print("Split checksum  :", split_checksum(data["labeled_idx"]), "(must be identical for everyone)")
