import torch
from torch.distributions.categorical import Categorical

from diffusion.diffusion import GaussianDiffusion
from diffusion.model import Unet


def get_input_size(task):
    if task=="sudoku":
        image_size = (9, 9)
        channels = 9
    elif task=="maze":
        image_size = (2*cfg.config.size+1, 2*cfg.config.size+1)
        channels = 2
    elif task=="grid":
        image_size = (1, 40)
        channels = 1
    elif task=="sushi":
        image_size = (1, 10)
        channels = 10
    elif task=="warcraft":
        image_size = (12, 12)
        channels = 2
    return image_size, channels

def get_model(cfg, device):
    image_size, channels = get_input_size(cfg.config.task)
    
    noise_model = Unet(
        dim = cfg.config.dims,
        init_dim = cfg.config.dims,
        self_condition = cfg.config.cond,
        channels = channels,
        device = device
    )
    
    dm = GaussianDiffusion(
        model = noise_model,
        timesteps = cfg.num_train_timesteps,
        image_size = image_size,
        device = device
    )

    return dm

def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim