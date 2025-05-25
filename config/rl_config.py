import os
import argparse
import pickle
import torch
import pprint

maze_samples = {
    5 : (64, 8, 16, 40),
    10 : (8, 1, 2, 5),
    15: (4, 1, 1, 2),
    20 : (2, 0, 1, 1)
}

class RL_Config:
    def __init__(self, args):
        # sample params ⬇
        self.sample_num_batches_per_epoch = 1
        self.train_num_inner_epochs = 1
        self.init_sample_num = 64
        
        if (args.task=="sudoku" and args.dataset=="minimal_17") or args.task=="grid":
            self.sample_batch_size = 32
            self.dynamic_sample_num = [256]*4+[128]*8+[64]*20
        elif args.task=="maze":
            self.sample_batch_size = maze_samples[args.size][0]
            self.dynamic_sample_num = [256] * maze_samples[args.size][1] + [128] * maze_samples[args.size][2] + [64] * maze_samples[args.size][3]
        else:
            self.sample_batch_size = 64
            self.dynamic_sample_num = [256]*8+[128]*16+[64]*40
        # train params ⬇
        self.save_freq_step = 20
        self.save_freq = 1
        self.patience = 5
        self.num_train_timesteps = args.timestep
        # optimizer paramas ⬇
        self.train_epoch_num = 100
        self.train_learning_rate = 1e-6
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
        self.name = "Reward"
        self.reward = 'board' # only for Sudoku

        args.cuda = args.cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        args.device = device
        dtype = torch.float32
        
        # read config
        self.config = args
        self.device = device
        self.ckpt_path = "" # set your own
        self.ckpt_model = "" # set your own
        

    def to_dict(self):
        # Convert the attributes of this class to a dictionary
        config_dict = {
            'sample_batch_size': self.sample_batch_size,
            'sample_num_batches_per_epoch': self.sample_num_batches_per_epoch,
            'train_num_inner_epochs': self.train_num_inner_epochs,
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
            'dataset': self.config.dataset,
            'split': self.split
        }
        return config_dict
