import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import get_list, png_to_jpeg
from .mscoco import MSCOCO2017
from .flickr import Flickr30k


class TrainDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, add_jpeg=True):
        assert split in ["train", "val"]

        # Load the dataset for training
        real_list = get_list(os.path.join(data_path, "mscoco2017", f"{split}2017"))
        fake_list = get_list(os.path.join(data_path, "stable-diffusion-v1-4", f"{split}2017"))

        # Setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        # Construct the entire dataset
        self.total_list = real_list + fake_list
        np.random.shuffle(self.total_list)

        # JPEG compression
        self.add_jpeg = add_jpeg

        # Transformations
        self.transform = transform

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        image = Image.open(img_path).convert("RGB")

        # Add JPEG compression
        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        # Apply the transformation
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, dataset, model, root_path, transform=None, add_jpeg=True):
        # Load fake images (.png và .jpg)
        fake_dir = os.path.join(root_path, dataset, model)
        self.fake = sorted([
            os.path.join(fake_dir, i)
            for i in os.listdir(fake_dir)
            if i.endswith(".png") or i.endswith(".jpg")
        ])

        # Load real images (.png và .jpg)
        real_dir = os.path.join(root_path, dataset, "real")
        if not os.path.exists(real_dir):
            raise ValueError(f"Real images directory not found: {real_dir}")

        self.real = sorted([
            os.path.join(real_dir, i)
            for i in os.listdir(real_dir)
            if i.endswith(".png") or i.endswith(".jpg")
        ])

        # Giữ nguyên chức năng cũ
        self.image_idx = list(range(len(self.real) + len(self.fake)))
        self.labels = [0] * len(self.real) + [1] * len(self.fake)

        # ⭐ Thêm chức năng (không ảnh hưởng code cũ)
        self.image_paths = self.real + self.fake

        self.add_jpeg = add_jpeg
        self.transform = transform

    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, idx):
        if idx < len(self.real):
            img_path = self.real[idx]
        else:
            img_path = self.fake[idx - len(self.real)]

        image = Image.open(img_path).convert("RGB")

        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label, img_path


import csv

csv_dir = os.path.join(args.save_dir, "csv_outputs")
os.makedirs(csv_dir, exist_ok=True)

csv_path = os.path.join(csv_dir, f"{dataset_name}_{model_name}.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path_to_image", "true_label", "pred_percentage", "pred_label"])

    idx_global = 0
    for images, labels, paths in tqdm(test_loader, desc="Saving CSV"):
        preds = detector.predict(images)

        for i in range(len(preds)):
            pred_score = float(preds[i])
            pred_label = 1 if pred_score > 0.5 else 0

            writer.writerow([
                paths[i],
                int(labels[i]),
                pred_score,
                pred_label
            ])

            idx_global += 1

print(f"Saved CSV: {csv_path}")


