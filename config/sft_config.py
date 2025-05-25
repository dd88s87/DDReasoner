import os
import argparse
import pickle
import torch
import pprint

class SFT_Config:
    def __init__(self, args):
        # train params ⬇
        self.train_batch_size = 1024
        self.save_freq = 100
        self.num_train_timesteps = args.timestep
        # optimizer paramas ⬇
        self.train_epoch_num = 5000
        self.train_learning_rate = 3e-5
        self.train_adam_beta1 = 0.9
        self.train_adam_beta2 = 0.999
        self.train_adam_weight_decay = 1e-4
        self.train_adam_epsilon = 1e-8
        # loss paramas ⬇
        self.train_adv_clip_max = 1.0
        self.train_clip_range = 0.2
        # seed ⬇
        self.seed = 3407
        # name
        self.name = "SFT"

        args.cuda = args.cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        args.device = device
        dtype = torch.float32
        
        # read config
        self.config = args
        self.device = device

    def to_dict(self):
        # Convert the attributes of this class to a dictionary
        config_dict = {
            'train_batch_size': self.train_batch_size,
            'save_freq': self.save_freq,
            'num_train_timesteps': self.num_train_timesteps,
            'train_learning_rate': self.train_learning_rate,
            'train_adam_beta1': self.train_adam_beta1,
            'train_adam_beta2': self.train_adam_beta2,
            'train_adam_weight_decay': self.train_adam_weight_decay,
            'train_adam_epsilon': self.train_adam_epsilon,
            'train_adv_clip_max': self.train_adv_clip_max,
            'train_clip_range': self.train_clip_range,
            'seed': self.seed,
            'config': vars(self.config),  # Make sure config is also a dictionary
            'device': str(self.device),  # Convert device to string for better readability
            'dataset': self.config.dataset
        }
        return config_dict
