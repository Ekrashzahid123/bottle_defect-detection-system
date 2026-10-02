import os
import glob
import json
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

CLASS_NAMES = ["Normal", "Defective"]

def parse_yolo_labels(label_path: str) -> int:
    """
    Parses YOLO label text file.
    Rule:
    - If ANY detected object has class 1 (open-cap / defect), the bottle is DEFECTIVE (1).
    - If all detected objects are class 0 (Good-cap), the bottle is NORMAL (0).
    - If empty or invalid, returns -1.
    """
    if not os.path.exists(label_path):
        return -1
        
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    if not lines:
        return -1
        
    classes = []
    for line in lines:
        parts = line.split()
        if parts:
            try:
                classes.append(int(parts[0]))
            except ValueError:
                pass
                
    if not classes:
        return -1
        
    # Multi-object rule: if any defect is present, the product is defective
    if 1 in classes:
        return 1
    elif all(c == 0 for c in classes):
        return 0
    else:
        return 0

def load_and_split_dataset(dataset_root: str = "dataset",
                           test_size: float = 0.15,
                           val_size: float = 0.15,
                           seed: int = 42) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    """
    Loads all annotated images across the dataset and performs a LEAK-FREE Group-Aware Stratified Split.
    All augmented versions of the same physical bottle (base_id) are strictly kept within the SAME split.
    No labels are inferred from filenames — 100% parsed from YOLO annotations.
    """
    all_image_paths = []
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_root, split, "images")
        if os.path.exists(img_dir):
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]:
                all_image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

    samples = []
    for img_path in sorted(all_image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Extract base physical image ID (e.g., 'IMG_1100' from 'IMG_1100_jpg.rf.803fa...')
        base_id = base_name.split(".rf.")[0].replace("_jpg", "").replace("_JPG", "")
        
        # Check label path in train or valid labels folder
        label_found = False
        label = -1
        for s in ["train", "valid", "test"]:
            lbl_candidate = os.path.join(dataset_root, s, "labels", f"{base_name}.txt")
            if os.path.exists(lbl_candidate):
                label = parse_yolo_labels(lbl_candidate)
                if label in [0, 1]:
                    label_found = True
                    break
                    
        if label_found and label in [0, 1]:
            samples.append({
                "path": img_path,
                "base_id": base_id,
                "label": label
            })

    if not samples:
        raise ValueError(f"No valid labeled samples found in {dataset_root}")

    paths = np.array([s["path"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    groups = np.array([s["base_id"] for s in samples])

    # 1. First Group Split: Separate Holdout Test Set (e.g. 15%)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(paths, labels, groups=groups))

    tv_paths, tv_labels, tv_groups = paths[train_val_idx], labels[train_val_idx], groups[train_val_idx]
    test_paths, test_labels = paths[test_idx], labels[test_idx]

    # 2. Second Group Split: Separate Validation Set (e.g. 15% / 85% ~ 17.6%)
    val_ratio = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss_val.split(tv_paths, tv_labels, groups=tv_groups))

    train_paths, train_labels = tv_paths[train_idx], tv_labels[train_idx]
    val_paths, val_labels = tv_paths[val_idx], tv_labels[val_idx]

    # Verify zero leakage
    train_bases = set(tv_groups[train_idx])
    val_bases = set(tv_groups[val_idx])
    test_bases = set(groups[test_idx])

    assert len(train_bases.intersection(val_bases)) == 0, "Data leakage between train and validation!"
    assert len(train_bases.intersection(test_bases)) == 0, "Data leakage between train and test!"
    assert len(val_bases.intersection(test_bases)) == 0, "Data leakage between validation and test!"

    split_paths = {
        "train": train_paths.tolist(),
        "valid": val_paths.tolist(),
        "test": test_paths.tolist()
    }
    split_labels = {
        "train": train_labels.tolist(),
        "valid": val_labels.tolist(),
        "test": test_labels.tolist()
    }

    return split_paths, split_labels

def create_tf_dataset(filepaths: List[str], labels: List[int], img_size: int = 224, batch_size: int = 32, is_training: bool = True):
    def load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)
        
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(factor=0.2),
            tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05)),
        ], name="industrial_data_augmentation")
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_data_loaders(dataset_root: str = "dataset", batch_size: int = 32, img_size: int = 224, seed: int = 42):
    split_paths, split_labels = load_and_split_dataset(dataset_root, test_size=0.15, val_size=0.15, seed=seed)

    train_ds = create_tf_dataset(split_paths["train"], split_labels["train"], img_size, batch_size, is_training=True)
    valid_ds = create_tf_dataset(split_paths["valid"], split_labels["valid"], img_size, batch_size, is_training=False)
    test_ds = create_tf_dataset(split_paths["test"], split_labels["test"], img_size, batch_size, is_training=False)

    # Class Weights for loss balancing
    tr_lbls = split_labels["train"]
    n_normal = sum(1 for l in tr_lbls if l == 0)
    n_defective = sum(1 for l in tr_lbls if l == 1)
    total = len(tr_lbls)

    class_weight = {
        0: (total / (2.0 * n_normal)) if n_normal > 0 else 1.0,
        1: (total / (2.0 * n_defective)) if n_defective > 0 else 1.0
    }

    return train_ds, valid_ds, test_ds, class_weight, split_paths, split_labels
