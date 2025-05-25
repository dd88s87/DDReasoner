import os
import argparse
import pickle
import torch
import pprint

class Test_Config:
    def __init__(self, args):
        self.num_train_timesteps = args.timestep

        args.cuda = args.cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        args.device = device
        dtype = torch.float32
        
        self.config = args
        self.device = device
        self.ckpt_path = "" # set your own
        

    def to_dict(self):
        # Convert the attributes of this class to a dictionary
        config_dict = {
            'num_train_timesteps': self.num_train_timesteps,
            'config': vars(self.config),  # Make sure config is also a dictionary
            'device': str(self.device),  # Convert device to string for better readability
            'dataset': self.config.dataset,
            'ckpt_path': self.ckpt_path
        }
        return config_dict
