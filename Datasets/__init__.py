from .dataset import TrainDataset, TestDataset

# List of evaluated real datasets
# Note: Please download the other datasets [cc3m, textcaps, sbu] from their original sources
EVAL_DATASET_LIST = ['real_coco_valid', 'real_ffhq']

# List of evaluated generative models
EVAL_MODEL_LIST = [
    # CompVis
    "DDPM",
    "Deepfloyd-IF_Stage_III",
    # runwayml
    "DiT_256",
    # segmind
    "DiT_512",
]

__all__ = ["TrainDataset", "TestDataset", "EVAL_DATASET_LIST", "EVAL_MODEL_LIST"]
