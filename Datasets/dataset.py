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
        fake_dir = os.path.join(root_path, dataset, model)
        self.fake = sorted([
            os.path.join(fake_dir, i)
            for i in os.listdir(fake_dir)
            if i.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        real_dir = os.path.join(root_path, dataset, "real")
        if not os.path.exists(real_dir):
            raise ValueError(f"Real images directory not found: {real_dir}")

        self.real = sorted([
            os.path.join(real_dir, i)
            for i in os.listdir(real_dir)
            if i.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.image_idx = list(range(len(self.real) + len(self.fake)))
        self.labels = [0] * len(self.real) + [1] * len(self.fake)
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

        # ---- FIX: Bỏ qua ảnh hỏng / lỗi ----
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print("Lỗi ảnh hỏng:", img_path)
            # load ảnh kế tiếp thay thế
            return self.__getitem__((idx + 1) % len(self))

        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label, img_path
