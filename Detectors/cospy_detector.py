import torch
import random
from torchvision import transforms
from utils import data_augment, weights2cpu
from .semantic_detector import SemanticDetector
from .artifact_detector import ArtifactDetector


# CO-SPY Detector
class CospyDetector(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(CospyDetector, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector()
        self.sem_dim = self.sem.fc.in_features

        # Load the artifact detector
        self.art = ArtifactDetector()
        self.art_dim = self.art.fc.in_features

        # Classifier
        self.fc = torch.nn.Linear(self.sem_dim + self.art_dim, num_classes)

        # Transformations inside the forward function
        # Including the normalization and resizing (only for the artifact detector)
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            flip_func,
            aug_func,
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

    def forward(self, x, dropout_rate=0.3):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)

        # Forward pass
        sem_feat, sem_coeff = self.sem(x_sem, return_feat=True)
        art_feat, art_coeff = self.art(x_art, return_feat=True)

        # Dropout
        if self.train():
            # Random dropout
            if random.random() < dropout_rate:
                # Randomly select a feature to drop
                idx_drop = random.randint(0, 1)
                if idx_drop == 0:
                    sem_coeff = torch.zeros_like(sem_coeff)
                else:
                    art_coeff = torch.zeros_like(art_coeff)

        # Concatenate the features
        x = torch.cat([sem_coeff * sem_feat, art_coeff * art_feat], dim=1)
        x = self.fc(x)

        return x
    # --- Save checkpoint (model + optimizer + epoch) ---
    def save_checkpoint(self, path, optimizer=None, epoch=0):
        ckpt = {
            "sem": self.sem.state_dict(),
            "art": self.art.state_dict(),
            "classifier": self.fc.state_dict(),
            "epoch": epoch,
        }

        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()

        torch.save(ckpt, path)


    # --- Load checkpoint ---
    def load_checkpoint(self, path, optimizer=None):
        ckpt = torch.load(path, map_location="cpu")

        # Load all submodules
        self.sem.load_state_dict(ckpt["sem"])
        self.art.load_state_dict(ckpt["art"])
        self.fc.load_state_dict(ckpt["classifier"])

        # Load optimizer nếu có
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        # Trả epoch để train tiếp
        return ckpt.get("epoch", 0)

# Define the label smoothing loss
class LabelSmoothingBCEWithLogits(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCEWithLogits, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        target = target.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss sửa cái nào
