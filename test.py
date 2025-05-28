from symbolic.args import init_argparse
from symbolic.models import get_model
from config.test_config import Test_Config

from symbolic import dataset

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import networkx as nx

def compute_rewards_sudoku(x, solutions, mask, dataset):
    '''
    use correctness as reward fuction
    '''
    n_samples = x.shape[0]
    x_softmax = F.softmax(x, dim=1)
    pred = torch.argmax(x_softmax, dim=1) + 1

    rewards = []
    correct_rates = []
    
    for i in range(0, n_samples):
        gold = solutions[i]
        m = mask[i][0]
        
        total_num = 0
        correct_num = 0
        for row in range(9):
            for col in range(9):
                if m[row, col] == False:
                    total_num+=1
                    if pred[i][row, col]==gold[row, col]:
                        correct_num+=1

        correct_rate = correct_num / total_num
        
        if dataset!="multiple_sol":
            reward = int(torch.equal(pred[i], gold))
        else:
            row_tag = True
            for row in range(9):
                row_tag = row_tag and (True if torch.equal(torch.unique(pred[i][row]), torch.tensor(range(1, 10)).to(pred[i][row].device)) else False)
                
            col_tag = True
            for col in range(9):
                board_col = torch.stack([pred[i][row,col] for row in range(9)])
                col_tag = col_tag and (True if torch.equal(torch.unique(board_col), torch.tensor(range(1, 10)).to(board_col.device)) else False)

            cube_tag = True
            for idx in range(9):
                board_cube = torch.stack([pred[i][row,col] for row in range((idx//3)*3,(idx//3)*3+3) for col in range((idx%3)*3,(idx%3)*3+3)])
                cube_tag = cube_tag and (True if torch.equal(torch.unique(board_cube), torch.tensor(range(1, 10)).to(board_cube.device)) else False)
            
            reward = int(row_tag and col_tag and cube_tag)
        
        rewards.append(reward)
        correct_rates.append(correct_rate)
    
    print("Rewards:", np.mean(rewards), "Correct rates:", np.mean(correct_rates))
    
    return rewards, correct_rates

def compute_rewards_maze(x, solutions):
    '''
    use correctness as reward fuction
    '''
    n_samples = x.shape[0]
    pred = torch.argmax(x, dim=1)
    gold = torch.argmax(solutions, dim=1)

    rewards = []
    
    for i in range(0, n_samples):
        reward = 1 if torch.equal(pred[i], gold[i]) else 0
        rewards.append(reward)
    
    print("Rewards:", np.mean(rewards))
    return rewards

# check connectivity of Warcraft map
def is_connected_warcraft(grid):
    H, W = grid.shape
    visited = torch.zeros_like(grid, dtype=torch.bool)
    
    directions = [  # 8 directions
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    
    def dfs(x, y):
        if not (0 <= x < H and 0 <= y < W):
            return
        if visited[x, y] or grid[x, y] != 1:
            return
        visited[x, y] = True
        for dx, dy in directions:
            dfs(x + dx, y + dy)

    if grid[0, 0] != 1 or grid[11, 11] != 1:
        return False

    dfs(0, 0)
    return visited[11, 11].item()

def compute_rewards_warcraft(x, solutions):
    '''
    use correctness as reward fuction
    '''
    n_samples = x.shape[0]
    pred = torch.sign(x[:,1])

    rewards = []
    hammings = []
    consistents = []
    
    for i in range(0, n_samples):
        consistent = 1 if is_connected_warcraft((pred[i].int()+1)//2) else 0
        reward = 1 if torch.equal(pred[i], solutions[i]) else int((consistent==1 and torch.equal(x[i,0] * pred[i].sum().item(), x[i,0] * solutions[i].sum().item())))
        rewards.append(reward)
        consistents.append(consistent)
    
    print("Rewards:", np.mean(rewards), "Consistents:", np.mean(consistents))
    return rewards, consistents

# check connectivity of Grid
edge_list = [
    (0,1), (1,2), (2,3),
    (4,5), (5,6), (6,7),
    (8,9), (9,10), (10,11),
    (12,13), (13,14), (14,15),
    (0,4), (4,8), (8,12), (1,5),
    (5,9), (9,13), (2,6), (6,10),
    (10,14), (3,7), (7,11), (11,15)
]

def is_connected_grid(predicted_edge_ids, source, target):
    predicted_edges = [edge_list[i] for i in predicted_edge_ids]
    subG = nx.Graph()
    subG.add_edges_from(predicted_edges)

    return nx.is_connected(subG) and source in subG and target in subG

def compute_rewards_grid(x, solutions):
    '''
    use correctness as reward fuction
    '''
    n_samples = x.shape[0]
    pred = torch.sign(x)

    rewards = []
    hammings = []
    consistents = []
    
    for i in range(0, n_samples):
        reward = 1 if torch.equal(pred[i], solutions[i]) else 0
        hamming = (pred[i][0,0,:24] == solutions[i][0,0,:24]).float().sum().item() / 24
        pred_edges = torch.nonzero(pred[i][0,0,:24] == 1.0, as_tuple=True)[0]
        src_dst = torch.nonzero(pred[i][0,0,24:] == 1.0, as_tuple=True)[0]
        consistent = int(is_connected_grid(list(pred_edges), list(src_dst)[0].item(), list(src_dst)[1].item())) if list(pred_edges)!=[] else 0
                
        rewards.append(reward)
        hammings.append(hamming)
        consistents.append(consistent)
    
    print("Rewards:", np.mean(rewards), "Hammings:", np.mean(hammings), "Consistents:", np.mean(consistents))
    return rewards, hammings, consistents

def compute_rewards_sushi(x, solutions):
    '''
    use correctness as reward fuction
    '''
    
    n_samples = x.shape[0]
    pred = (torch.argmax(x, dim=1) + 1).squeeze(1)
    
    rewards = []
    correct_rates = []
    consistents = []
    
    for i in range(0, n_samples):
        reward = 1 if torch.equal(pred[i], solutions[i]) else 0
        rewards.append(reward)
        correct = (pred[i] == solutions[i]).float().mean().item()
        correct_rates.append(correct)
        consistent = 1 if torch.equal(torch.unique(pred[i]), torch.tensor(range(1, 11)).to(pred[i].device)) else 0
        consistents.append(consistent)
    
    print("Rewards:", np.mean(rewards), "Correct rates:", np.mean(correct_rates), "Consistents:", np.mean(consistents))
    return rewards, correct_rates, consistents
    

if __name__ == "__main__":
    # config
    args = init_argparse()
    config = Test_Config(args)
    datasets = dataset.initialize_datasets(config.config)
    
    # load model
    flow = get_model(config, config.device)
    flow.to(config.device)
    fn = 'generative_model.npy' 
    flow_state_dict = torch.load(config.ckpt_path+fn, map_location=config.device)
    flow.load_state_dict(flow_state_dict)
    flow.eval()
    
    # load data
    test_data = datasets['test']
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)
    total_reward = []
    total_correct = []
    total_consistent = []
    print("Dataset Len: ", len(test_data))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = {key: value.to(config.device) for key, value in batch.items()}
            puzzle = batch['puzzle']
            solution = batch['sol'] if config.config.task!="maze" and config.config.task!="grid" else None
            solution_onehot = batch['sol_onehot']
            mask = batch['mask']
            
            x, latents, logps, timestep, mu, sigma = flow.sample(puzzle.shape[0], mask = mask, context = puzzle, is_training=False)
            if config.config.task=="sudoku":
                rewards, correct_rates = compute_rewards_sudoku(x, solution, mask, config.config.dataset)
                consistents = rewards
            elif config.config.task=="maze":
                rewards = compute_rewards_maze(x, solution_onehot)
                correct_rates = rewards
                consistents = rewards
            elif config.config.task=="warcraft":
                rewards, correct_rates = compute_rewards_warcraft(x, solution)
                consistents = correct_rates
            elif config.config.task=="grid":
                rewards, correct_rates, consistents = compute_rewards_grid(x, solution_onehot)
            elif config.config.task=="sushi":
                rewards, correct_rates, consistents = compute_rewards_sushi(x, solution)
            total_reward.extend(rewards)
            total_correct.extend(correct_rates)
            total_consistent.extend(consistents)
    
    print(config.config.dataset, config.ckpt_path, "avg reward:", np.mean(total_reward), "  avg correct rates:", np.mean(total_correct), "  avg consistents:", np.mean(total_consistent))