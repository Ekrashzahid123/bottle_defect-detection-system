import os
import glob
from typing import Tuple, List, Dict
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["Normal", "Defective"]
RAW_TO_CLASS = {
    0: 0,  # Good-cap-Df0S -> Normal
    1: 1   # open-cap -> Defective
}

def load_split_filepaths_and_labels(dataset_root: str, split: str) -> Tuple[List[str], List[int]]:
    images_dir = os.path.join(dataset_root, split, "images")
    labels_dir = os.path.join(dataset_root, split, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    valid_images = [f for f in image_files if os.path.splitext(f)[1] in image_extensions]

    filepaths = []
    labels = []

    for img_path in valid_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                content = f.read().strip()
            if content:
                raw_cls = int(content.split()[0])
                if raw_cls in RAW_TO_CLASS:
                    filepaths.append(img_path)
                    labels.append(RAW_TO_CLASS[raw_cls])
        elif split == "test":
            # For test split default to Normal or placeholder
            filepaths.append(img_path)
            labels.append(0)

    return filepaths, labels

def create_tf_dataset(filepaths: List[str], labels: List[int], img_size: int = 224, batch_size: int = 32, is_training: bool = True):
    def load_and_preprocess_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)
        
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Data Augmentation layer in TF
    if is_training:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomZoom(0.1),
        ], name="data_augmentation")
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_tf_data_loaders(dataset_root: str = "dataset", batch_size: int = 32, img_size: int = 224):
    train_paths, train_labels = load_split_filepaths_and_labels(dataset_root, "train")
    valid_paths, valid_labels = load_split_filepaths_and_labels(dataset_root, "valid")
    test_paths, test_labels = load_split_filepaths_and_labels(dataset_root, "test")

    train_ds = create_tf_dataset(train_paths, train_labels, img_size, batch_size, is_training=True)
    valid_ds = create_tf_dataset(valid_paths, valid_labels, img_size, batch_size, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_labels, img_size, batch_size, is_training=False)

    # Compute balanced class weights for Keras model.fit
    n_normal = sum(1 for l in train_labels if l == 0)
    n_defective = sum(1 for l in train_labels if l == 1)
    total = len(train_labels)

    class_weight = {
        0: (total / (2.0 * n_normal)) if n_normal > 0 else 1.0,
        1: (total / (2.0 * n_defective)) if n_defective > 0 else 1.0
    }

    return train_ds, valid_ds, test_ds, class_weight, (train_paths, valid_paths, test_paths)
