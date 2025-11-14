from .dataset import TrainDataset, TestDataset

# List of evaluated real datasets
# Note: Please download the other datasets [cc3m, textcaps, sbu] from their original sources
EVAL_DATASET_LIST = [
    "real_coco_valid",
    "real_ffhq",
    "real_lsun"
]

# Danh sách model generative
EVAL_MODEL_LIST = [
    "DDPM",
    "Deepfloyd-IF_Stage_III",
    "stargan",
    "latent-diffusion_noise2image_LSUNbedrooms",
    "gaugan",
    "latent-diffusion_text2img_valid",
    "guided-diffusion_noise2image_LSUNhorses",
    "GigaGAN_cond_imagenet256",
    "progan_lsun",
    "galip_cocoval",
    "dalle3_cocoval",
    "DiffusionProjectedGan_lsunbed_256",
    "stable_diffusion_2_1_768",
    "GigaGAN_t2i_coco256_diff_noised",
    "stylegan2_ffhq_1024x1024",
    "sdxl_cocoval",
    "GigaGAN_t2i_coco256_rep",
    "latent-diffusion_noise2image_FFHQ",
    "DiT_256",
    "stylegan3_r",
    "stylegan2_church",
    "stylegan2_lsundog_256x256",
    "dalle_2",
    "biggan_256",
    "guided-diffusion_noise2image_LSUNcats",
    "stable_diffusion_2_1_512",
    "biggan_512",
    "latent-diffusion_class2image_ImageNet",
    "guided-diffusion_class2image_ImageNet",
    "stable_diffusion_256",
    "latent-diffusion_noise2image_LSUNchurches",
    "stylegan3_t",
    "DiffusionProjectedGan_lsunchurch_256",
    "glide_text2img_valid",
    "stable_diffusion_2_1_256",
    "stylegan2_ffhq_256x256",
    "DiT_512",
    "guided-diffusion_noise2image_LSUNbedrooms"
]

__all__ = ["TrainDataset", "TestDataset", "EVAL_DATASET_LIST", "EVAL_MODEL_LIST"]
