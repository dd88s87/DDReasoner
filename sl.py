from trainer.sl_trainer import SLTrainer
from tqdm import tqdm as tq
import os
import wandb
import torch
import numpy
import random
import pprint

def set_seed(seed):
    random.seed(seed)      
    torch.manual_seed(seed) 
    numpy.random.seed(seed) 

if __name__ == "__main__":
    # trainer init
    trainer = SLTrainer()
    # set seed
    set_seed(trainer.config.seed)
    pprint.pprint(trainer.config.to_dict())
    # wandb init
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "" # set your own
    
    global_step = 0
    min_loss = 10000
    wandb.init(project="sft", name=trainer.config.name, config=trainer.config.to_dict())
    wandb.save('*.txt')
    if trainer.config.config.task=="sudoku":
        root_path = os.path.join("exp", trainer.config.config.dataset+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate))
    elif trainer.config.config.task=="maze":
        root_path = os.path.join("exp", "maze"+str(trainer.config.config.size)+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate))
    elif trainer.config.config.task=="warcraft":
        root_path = os.path.join("warcraft", trainer.config.config.task+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate)+"_size"+str(trainer.config.config.size))
    else:
        root_path = os.path.join("exp", trainer.config.config.task+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate))

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    for i in tq(range(0,trainer.config.train_epoch_num),desc = "Training epoch",unit = "epoch"):
        results = trainer.sft_step(i, global_step)
        
        if (i+1) % trainer.config.save_freq == 0:
            trainer.save_model(root_path, str(i))
        
        loss = []
        for result in results:
            loss.append(result['loss'])
            for key,value in result.items():
                wandb.log({key: value}, commit=True)
        if min_loss > numpy.mean(loss):
            min_loss = numpy.mean(loss)
            trainer.save_model(root_path, "best")
            