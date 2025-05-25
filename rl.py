from trainer.rl_trainer import RLTrainer
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
    trainer = RLTrainer()
    # set seed
    set_seed(trainer.config.seed)
    pprint.pprint(trainer.config.to_dict())
    # wandb init
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "" # set your own

    
    global_step = 0
    best_reward = 0
    patience_count = 0
    wandb.init(project="rl", name=trainer.config.name, config=trainer.config.to_dict())
    wandb.save('*.txt')
    
    if trainer.config.config.task=="sudoku":
        root_path = os.path.join("exp", trainer.config.config.dataset+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate)+"_SFT"+trainer.config.ckpt_model)
    elif trainer.config.config.task=="maze":
        root_path = os.path.join("exp", "maze"+str(trainer.config.config.size)+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate)+"_SFT"+trainer.config.ckpt_model)
    else:
        root_path = os.path.join("exp", trainer.config.config.task+"_"+trainer.config.name+"_"+str(trainer.config.train_learning_rate)+"_SFT"+trainer.config.ckpt_model)
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    for i in tq(range(0,trainer.config.train_epoch_num),desc = "Training epoch", unit = "epoch"):
        results, avg_acc = trainer.step(i,global_step,root_path)
        
        if (i+1) % trainer.config.save_freq == 0:
            trainer.save_model(root_path,"epoch"+str(i+1))
        
        for result in results:
            for key,value in result.items():
                wandb.log({key: value}, commit=True)
        
        if best_reward < avg_acc:
            best_reward = avg_acc
            patience_count = 0
            trainer.save_model(root_path, "best")
        else:
            patience_count += 1
            if patience_count >= trainer.config.patience:
                break