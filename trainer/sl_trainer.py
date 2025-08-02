import os
os.environ["OMP_NUM_THREADS"] = "18"

from config.sl_config import SL_Config
from symbolic import dataset
from symbolic.models import get_model
from symbolic.args import init_argparse

import torch
from trl.trainer import BaseTrainer
from os.path import join

from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class SLTrainer(BaseTrainer):
    """
    The SLTrainer to optimise diffusion models.
    """
    def __init__(self, ckpt_path=None):
        self.args = init_argparse()
        self.config = SL_Config(self.args)
        self.generate_model = self._create_pipline(ckpt_path)

        params = list(self.generate_model.parameters())
        self.optimizer = self._setup_optimizer(params)
        self.datasets = dataset.initialize_datasets(self.config.config)
        self.train_loader = DataLoader(self.datasets['train'], batch_size=self.config.train_batch_size, shuffle=True)
    
    def save_model(self,path,name):
        torch.save(self.generate_model.state_dict(), join(path,f"{name}_generative_model.npy"))
        torch.save(self.optimizer.state_dict(), join(path,f"{name}_optimize.npy"))
    
    def _create_pipline(self, ckpt_path=None):
        '''
        Load model parameters
        
        Returns:
            flow (torch.nn.moudle): The model
        '''
        config = self.config
            
        flow = get_model(config, config.device)
        flow.to(config.device)
        if ckpt_path is not None:
            fn = 'generative_model.npy'
            flow_state_dict = torch.load(ckpt_path+fn, map_location=config.device )
            flow.load_state_dict(flow_state_dict)
        
        return flow
    
    def sft_step(self, epoch: int, global_step: int):
        results = []
        print("epoch:", epoch, " data_len:", len(self.train_loader))

        for batch_idx, batch in enumerate(self.train_loader):
            # 1. prepare input
            batch = {key: value.to(self.config.device) for key, value in batch.items()}
            puzzle = batch['puzzle']
            solution_onehot = batch['sol_onehot']
            mask = batch['mask']
            
            # 2. train diffusion
            loss = self.generate_model(solution_onehot, mask = mask, context = puzzle)
            loss.backward()
            clip_grad_norm_(self.generate_model.parameters(), max_norm=1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            result = {}
            result['loss'] = loss.item()
            result["globalStep"] = global_step + 1
            results.append(result)
        
        return results
    
    def _setup_optimizer(self, trainable_layers_parameters):
        optimizer_cls = torch.optim.AdamW
        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )
