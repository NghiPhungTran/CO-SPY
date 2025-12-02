from .dataset import TrainDataset, TestDataset

# List of evaluated real datasets
EVAL_DATASET_LIST = [
    "real"    
]
# Danh sách model generative
EVAL_MODEL_LIST = [
    "stable_diffusion"
]
__all__ = ["TrainDataset", "TestDataset", "EVAL_DATASET_LIST", "EVAL_MODEL_LIST"]
