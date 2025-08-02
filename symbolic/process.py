from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F

import random


class PreProcessedSudoku(Dataset):
    """
    Data structure for a pre-processed dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, data, labels, num_pts=-1):
        self.data = data
        self.labels = labels

        if num_pts < 0:
            self.num_pts = data.shape[0]
        else:
            self.num_pts = num_pts

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        xt = F.one_hot(x,num_classes=10).permute(2,0,1)[1:].to(torch.float32)
        mask = x.ne(0)
        mask = mask.unsqueeze(0).repeat([9,1,1])
        y = torch.from_numpy(self.labels[idx])
        yt = F.one_hot(y,num_classes=10).permute(2,0,1)[1:].to(torch.float32)
        xt = xt * 2 - 1
        yt = yt * 2 - 1
        return {"puzzle": xt, "sol": y, "sol_onehot": yt, "mask": mask, "idx": idx}


class DynamicSample(Dataset):
    """
    Data structure for a dynamic dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, data, num_pts=-1):
        self.data = data

        if num_pts < 0:
            self.num_pts = data.shape[0]
        else:
            self.num_pts = num_pts

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        item = self.data[idx]
        return {key: value for key, value in item.items()}


class PreProcessedMaze(Dataset):
    """
    Data structure for a pre-processed dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, data, num_pts=-1):
        self.data = data

        if num_pts < 0:
            self.num_pts = data.shape[0]
        else:
            self.num_pts = num_pts

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        '''
        data:
        - map: (Nx, Ny, 2)
        - goal: (Nx, Ny, 3)
        - path: (Nx, Ny, 3)

        '''
        # get id
        maze_solved = torch.tensor(self.data[idx], dtype=torch.long)

        # get map, goal, path (solution)
        map = maze_solved[:,:,0]
        goal = maze_solved[:,:,1]   
        path = maze_solved[:,:,2]
        
        # sonvert the values to 0 and 1
        map = ((map+1)/2).long()
        goal = (goal+1).long()
        path = ((path+1)/2).long()
        mask = (goal != 0) | (map == 0)
        
        # one-hot encoding
        map_one_hot = F.one_hot(map, num_classes=2).permute(2,0,1).to(torch.float32)
        path_one_hot = F.one_hot(path, num_classes=2).permute(2,0,1).to(torch.float32)
        mask = mask.unsqueeze(0).repeat([2,1,1])
        map_one_hot = map_one_hot*2-1
        path_one_hot = path_one_hot*2-1

        return {"puzzle": map_one_hot, "sol_onehot": path_one_hot, "mask": mask, "idx": idx}
    

class PreProcessedMazeResize(Dataset):
    """
    Data structure for a pre-processed dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, data, large_size=10, small_size=5, num_pts=-1):
        self.data = data

        if num_pts < 0:
            self.num_pts = data.shape[0]
        else:
            self.num_pts = num_pts
            
        self.max_len = (large_size - small_size) * 2
        self.large_size = large_size
        self.small_size = small_size

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        '''
        data:
        - map: (Nx, Ny, 2)
        - goal: (Nx, Ny, 3)
        - path: (Nx, Ny, 3)

        '''
        # get id
        maze_solved = torch.tensor(self.data[idx], dtype=torch.long)

        # get map, goal, path (solution)
        map = maze_solved[:,:,0]
        goal = maze_solved[:,:,1]   
        path = maze_solved[:,:,2]
        
        # sonvert the values to 0 and 1
        map = ((map+1)/2).long()
        goal = (goal+1).long()
        path = ((path+1)/2).long()
        mask = (goal != 0) | (map == 0)
        large_map = torch.zeros((2*self.large_size+1, 2*self.large_size+1), dtype=int)
        large_path = torch.zeros((2*self.large_size+1, 2*self.large_size+1), dtype=int)
        large_mask = torch.ones((2*self.large_size+1, 2*self.large_size+1), dtype=bool)

        x = random.randint(0, self.max_len)
        y = random.randint(0, self.max_len)
        large_map[x:x+2*self.small_size+1, y:y+2*self.small_size+1] = map
        large_mask[x:x+2*self.small_size+1, y:y+2*self.small_size+1] = mask
        large_path[x:x+2*self.small_size+1, y:y+2*self.small_size+1] = path
        
        map_one_hot = F.one_hot(large_map, num_classes=2).permute(2,0,1).to(torch.float32)
        path_one_hot = F.one_hot(large_path, num_classes=2).permute(2,0,1).to(torch.float32)
        mask = large_mask.unsqueeze(0).repeat([2,1,1])  
        
        map_one_hot = map_one_hot*2-1
        path_one_hot = path_one_hot*2-1

        return {"puzzle": map_one_hot, "sol_onehot": path_one_hot, "mask": mask, "idx": idx}    

    
def to_one_hot(dense, n, inv=False):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    if inv:
        one_hot = (one_hot + 1) % 2
    return one_hot
    
class PreProcessedGrid(Dataset):
    """
    Data structure for a pre-processed dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, data, num_pts=-1):
        self.data = data

        if num_pts < 0:
            self.num_pts = data.shape[0]
        else:
            self.num_pts = num_pts

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        line = self.data[idx]
        tokens = line.strip().split(',')
        if(tokens[0] != ''):
            removed = [int(x) for x in tokens[0].split('-')]
        else:
            removed = []
            
        inp = [int(x) for x in tokens[1].split('-')]
        paths = tokens[2:]
        path = [int(x) for x in paths[0].split('-')]
        puzzle = np.concatenate((to_one_hot(removed, 24, True), to_one_hot(inp, 16))).reshape(1,1,40)*2-1
        solution = np.concatenate((to_one_hot(path, 24), to_one_hot(inp, 16))).reshape(1,1,40)*2-1
        mask = np.concatenate((to_one_hot(removed, 24, True)==0, np.array([True] * 16))).reshape(1,1,40)
        
        puzzle = torch.tensor(puzzle).to(torch.float32)
        solution = torch.tensor(solution).to(torch.float32)
        mask = torch.tensor(mask)
        
        return {"puzzle": puzzle, "sol_onehot": solution, "mask": mask, "idx": idx}
    
    
class PreProcessedSushi(Dataset):
    """
    Data structure for a pre-processed dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, data, num_pts=-1):
        self.data = data

        if num_pts < 0:
            self.num_pts = data.shape[0]
        else:
            self.num_pts = num_pts
        self.DATA_IND = [1, 2, 3, 5, 7, 8]
        self.LABEL_IND = [4, 6, 9, 10]
        self.general_mask = np.zeros((10, 10), dtype=bool)
        for row in self.DATA_IND:
            self.general_mask[row - 1] = True

    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, idx):
        line = self.data[idx]
        tokens = line.strip().split(',')
        ranking = [int(x) for x in tokens[1:]]
        solution = torch.tensor(np.array(ranking))
        xt = F.one_hot(solution, num_classes=11).unsqueeze(0).permute(2,0,1)[1:].to(torch.float32)
        
        mask = torch.ones((10, 1, 10), dtype=bool)
        for ind in self.LABEL_IND:
            mask[:, :, ranking.index(ind)] = False
        puzzle = (xt * mask)*2-1
        solution_onehot = xt * 2 - 1

        return {"puzzle": puzzle, "sol": solution, "sol_onehot": solution_onehot, "mask": mask, "idx": idx}
    
    
class PreProcessedWarcraft(Dataset):
    """
    Data structure for a pre-processed dataset.  Extends PyTorch Dataset.
    """
    def __init__(self, inputs, weights, labels, size, num_pts=-1):
        self.inputs = inputs
        self.weights = weights
        self.labels = labels
        self.size = size

        if num_pts < 0:
            self.num_pts = inputs.shape[0]
        else:
            self.num_pts = num_pts

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        # puzzle = 
        mean_ = self.weights[idx].mean()
        std_ = self.weights[idx].std()
        weights_norm = torch.tensor((self.weights[idx] - mean_) / std_)
        path = torch.tensor(self.labels[idx]).to(torch.float32)*2-1
        solution = torch.stack([weights_norm, path])
        mask = torch.zeros((2, self.size, self.size), dtype=bool)
        mask[0] = True
        mask[1, 0, 0]  = True
        mask[1, self.size-1, self.size-1] = True
        puzzle = solution * mask
        
        return {"puzzle": puzzle, "sol_onehot": solution, "sol": path, "mask": mask, "idx": idx}