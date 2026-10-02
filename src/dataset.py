import os
import glob
from typing import Tuple, List, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

CLASS_NAMES = ["Normal", "Defective"]
RAW_TO_CLASS = {
    0: 0,  # 0: Good-cap-Df0S -> Normal
    1: 1   # 1: open-cap -> Defective
}

def get_transforms(img_size: int = 224):
    """
    Returns data augmentation pipeline for training and standard normalization for validation/testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

class BottleDataset(Dataset):
    """
    PyTorch Dataset for Bottle Defect Detection.
    Parses images and corresponding YOLO-style annotation text files.
    """
    def __init__(self, root_dir: str, split: str = "train", transform=None, include_unlabeled: bool = False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        
        self._load_samples(include_unlabeled)

    def _load_samples(self, include_unlabeled: bool):
        images_dir = os.path.join(self.root_dir, self.split, "images")
        labels_dir = os.path.join(self.root_dir, self.split, "labels")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        image_files = glob.glob(os.path.join(images_dir, "*.*"))
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
        valid_images = [f for f in image_files if os.path.splitext(f)[1] in image_extensions]

        for img_path in valid_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    content = f.read().strip()
                if content:
                    # Class ID is the first token in YOLO format
                    raw_cls = int(content.split()[0])
                    if raw_cls in RAW_TO_CLASS:
                        self.samples.append((img_path, RAW_TO_CLASS[raw_cls]))
                elif include_unlabeled:
                    # Unlabeled images can be excluded or handled as requested
                    pass
            elif self.split == "test":
                # For test set if no labels provided, assign placeholder -1
                self.samples.append((img_path, -1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

    def get_class_distribution(self) -> Dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            if label in [0, 1]:
                counts[CLASS_NAMES[label]] += 1
        return counts

def get_data_loaders(dataset_root: str = "dataset",
                     batch_size: int = 32,
                     img_size: int = 224,
                     num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Creates train, valid, and test DataLoaders and computes class weights to handle imbalance.
    """
    train_tf, eval_tf = get_transforms(img_size)

    train_dataset = BottleDataset(dataset_root, split="train", transform=train_tf)
    valid_dataset = BottleDataset(dataset_root, split="valid", transform=eval_tf)
    test_dataset = BottleDataset(dataset_root, split="test", transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Calculate class weights for CrossEntropyLoss to address class imbalance
    dist = train_dataset.get_class_distribution()
    total = sum(dist.values())
    if total > 0 and dist["Normal"] > 0 and dist["Defective"] > 0:
        # Inverse frequency weighting
        weight_normal = total / (2.0 * dist["Normal"])
        weight_defective = total / (2.0 * dist["Defective"])
        class_weights = torch.tensor([weight_normal, weight_defective], dtype=torch.float32)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

    return train_loader, valid_loader, test_loader, class_weights
