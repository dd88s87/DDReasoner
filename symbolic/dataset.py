import torch
import numpy as np

import logging
import os

from symbolic.process import PreProcessedSudoku, PreProcessedMaze, PreProcessedMazeResize, PreProcessedGrid, PreProcessedSushi, PreProcessedWarcraft


def set_data(whole, num_train, num_val, num_test):
    train = np.array([whole[i] for i in range(num_train)])
    valid = np.array([whole[i] for i in range(num_train, num_train+num_val)])
    test = np.array([whole[i] for i in range(num_train+num_val, num_train+num_val+num_test)])
    return train, valid, test

SET_MAZEDATA_SIZE = 20000
# set your path of Maze data
data_list = {5: "",
             10: "",
             15: "",
             20: ""}

def get_warcraft_data(data_dir, split, ratio=None):
    if "12x12" in data_dir or split!="train":
        inputs = np.load(os.path.join(data_dir, split + "_maps.npy"))
        weights = np.load(os.path.join(data_dir, split + "_vertex_weights.npy"))
        labels = np.load(os.path.join(data_dir, split + "_shortest_paths.npy"))
    else:
        inputs = np.concatenate((np.load(os.path.join(data_dir, split + "_maps_part0.npy")), np.load(os.path.join(data_dir, split + "_maps_part1.npy"))), axis=0)
        weights = np.concatenate((np.load(os.path.join(data_dir, split + "_vertex_weights_part0.npy")), np.load(os.path.join(data_dir, split + "_vertex_weights_part1.npy"))), axis=0)
        labels = np.concatenate((np.load(os.path.join(data_dir, split + "_shortest_paths_part0.npy")), np.load(os.path.join(data_dir, split + "_shortest_paths_part1.npy"))), axis=0)
    if ratio:
        length = int(inputs.shape[0] * ratio)
        inputs, weights, labels = inputs[:length], weights[:length], labels[:length]
    return inputs, weights, labels

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def initialize_datasets(args):
    """
    Initialize datasets.
    """
    # Load and process dataset. Returns datafiles.
    if args.task=="sudoku":
        data_name = args.dataset
        data_path = "data/"+data_name+"/"+data_name+".npy"
        sol_path = "data/"+data_name+"/"+data_name+"_sol.npy"
        datafiles = np.load(data_path, allow_pickle=True).item()
        solfiles = np.load(sol_path, allow_pickle=True).item()
        
    elif args.task=="maze":
        grid_size = default(args.grid_size, args.size)
        data_path = f"data/maze/{data_list[grid_size]}"
        datafiles = []
        for i in range(SET_MAZEDATA_SIZE):
            datafiles.append(np.load(f"{data_path}/maze_solved-{i}.npy"))
        preprocesscls = PreProcessedMaze
    
    elif args.task=="grid":
        data_path = "data/grid/grid.data"
        datafiles = []
        with open(data_path) as file:
            for line in file:
                datafiles.append(line)
        preprocesscls = PreProcessedGrid
    
    elif args.task=="sushi":
        data_path = "data/sushi/sushi.soc"
        datafiles = []
        with open(data_path) as file:
            for line in file:
                tokens = line.strip().split(',')
                if len(tokens) >= 10: # Doesn't have enough entries, isn't data
                    datafiles.append(line)
        preprocesscls = PreProcessedSushi
    
    if args.task != "warcraft":
        data_length = len(datafiles)
        
        # Set the number of points based upon the arguments
        num_train = int(data_length * args.train_ratio)
        num_val = int(data_length * args.valid_ratio)
        num_test = int(data_length * args.test_ratio)
        
        # Split train/valid/test datasets
        train_data, valid_data, test_data = set_data(datafiles, num_train, num_val, num_test)
        
    datasets = {}
    
    # Process datasets
    if args.task=="sudoku":
        train_sol, valid_sol, test_sol = set_data(solfiles, num_train, num_val, num_test)
        datasets['train'] = PreProcessedSudoku(train_data, train_sol, num_pts=num_train)
        datasets['valid'] = PreProcessedSudoku(valid_data, valid_sol, num_pts=num_val)
        datasets['test'] = PreProcessedSudoku(test_data, test_sol, num_pts=num_test)
    elif (args.task=="maze" and args.resize==False) or args.task=="grid" or args.task=="sushi":
        datasets['train'] = preprocesscls(train_data, num_pts=num_train)
        datasets['valid'] = preprocesscls(valid_data, num_pts=num_val)
        datasets['test'] = preprocesscls(test_data, num_pts=num_test)
    elif args.task=="maze" and args.resize==True: 
        datasets['train'] = PreProcessedMazeResize(train_data, num_pts=num_train, large_size=args.size, small_size=grid_size)
        datasets['valid'] = PreProcessedMazeResize(valid_data, num_pts=num_val, large_size=args.size, small_size=grid_size)
        datasets['test'] = PreProcessedMazeResize(test_data, num_pts=num_test, large_size=args.size, small_size=grid_size)
        
    elif args.task == "warcraft":
        data_path = f'data/warcraft/warcraft_shortest_path_oneskin/{args.size}x{args.size}/'
        train_inputs, train_weights, train_labels = get_warcraft_data(data_path, "train", args.train_ratio)
        valid_inputs, valid_weights, valid_labels = get_warcraft_data(data_path, "val")
        test_inputs, test_weights, test_labels = get_warcraft_data(data_path, "test")
        
        datasets['train'] = PreProcessedWarcraft(train_inputs, train_weights, train_labels, args.size, num_pts=train_inputs.shape[0])
        datasets['valid'] = PreProcessedWarcraft(valid_inputs, valid_weights, valid_labels, args.size, num_pts=valid_inputs.shape[0])
        datasets['test'] = PreProcessedWarcraft(test_inputs, test_weights, test_labels, args.size, num_pts=test_inputs.shape[0])
    
    return datasets
