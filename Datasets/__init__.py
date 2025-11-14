from .dataset import TrainDataset, TestDataset

# List of evaluated real datasets
# Note: Please download the other datasets [cc3m, textcaps, sbu] from their original sources
EVAL_DATASET_LIST = [
    "laion"
]

# Danh sách model generative
EVAL_MODEL_LIST = [
    "DiffusionDB",
    "IF-CC1M",
    "SDv15R-CC1M",
    "stylegan3-t-ffhqu", "stylegan3-t-metfaces",
]

__all__ = ["TrainDataset", "TestDataset", "EVAL_DATASET_LIST", "EVAL_MODEL_LIST"]
