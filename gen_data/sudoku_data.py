from genericpath import isdir
from tabnanny import check
import numpy as np
import time
import statistics
import os
from tqdm import tqdm
from sudoku_solver.board import Board
import torch

import random
import shutil
import multiprocessing as mp
import argparse

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')
    
def init_parser():
    parser = argparse.ArgumentParser()
    # General args
    parser.add_argument('--solver', type=str, default='default',
                        help='symbolic solver to use. available options are default, prolog and backtrack')
    parser.add_argument('--dataset', type=str, default='all',
                        help='dataset to generate between [multiple_sol,minimal_17,big_kaggle,satnet_data,all]')
    return parser



def process_satnet_data(args, noise_input=False):
    print('----------------------------------------------')
    print('Processing dataset satnet')
    print('----------------------------------------------')
    data_name = 'satnet'

    if not isdir('data/'+data_name):
        os.mkdir('data/'+data_name)
    
    final_data, final_solutions = format_conversion_satnet()


def format_conversion_satnet():
    with open('data/original_data/features.pt', 'rb') as f:
            X_in = torch.load(f)
    with open('data/original_data/labels.pt', 'rb') as f:
        Y_in = torch.load(f)

    data_s, label, _ = process_inputs(X_in, Y_in)

    final_data = {}
    for idx, dat in enumerate(data_s):
        symbolic = dat.reshape([81,9])
        a = torch.zeros((1,81))
        symbolic = torch.cat((torch.t(a),symbolic),1)
        for i in range(81):
            if 1 not in symbolic[i]:
                symbolic[i][0]= 1.
        symbolic = symbolic.argmax(dim=1)
        symbolic = symbolic.unsqueeze(dim=0)
        final_data[idx] = symbolic.numpy().astype(int).reshape(9,9)

    final_labels = {}
    for idx, ll in enumerate(label):
        symbolic_label = ll.reshape([81,9])
        symbolic_label = symbolic_label.argmax(dim=1)+1
        symbolic_label = symbolic_label.unsqueeze(dim=0)
        final_labels[idx] = symbolic_label.numpy().astype(int).reshape(9,9)

    return final_data, final_labels


def dataset_generation(data_name,solver):
    print('Generating solutions')
    stats = []
    data_in = f'data/{data_name}/{data_name}.npy'
    data_out_sol = f'data/{data_name}/{data_name}_sol'
    solutions = {}
    counter = 0
    prolog_instance = None
    if solver == 'prolog':
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
    boards_dict = np.load(data_in,allow_pickle=True).item()
    for key in tqdm(boards_dict):
            board = Board(boards_dict[key])
            time1 = time.time()
            board.solve(solver, prolog_instance)
            time2 = time.time()
            solutions[counter] = board.board
            stats.append(time2 - time1)
            counter += 1
    np.save(data_out_sol, solutions)
    print('sudoku solved: ',len(stats))
    print(f'tot time: {sum(stats):4f}')
    print(f'mean time:  {statistics.mean(stats):4f}')
    print(f'max time:  {max(stats):4f}')
    print(f'min time:  {min(stats):4f}')    

def format_conversion(data_name,data_new_name):
    '''
    limit: 100000 data points
    '''
    print('Converting input format')
    data_in = f'data/original_data/{data_name}' 
    data_out = f'data/{data_new_name}/{data_new_name}'
    file_in = open(data_in, 'r')
    lines = file_in.readlines()
    data = {}
    data_list = []
    for line in tqdm(lines):
        if '#' not in line and len(line)>80:
            input_line = line.replace('.','0').replace('\n','')
            input_line = np.array([int(i) for i in input_line])
            input_line = input_line.reshape(9,9)
            data_list.append(input_line)
    file_in.close()
    # shuffle the dataset
    indices = [i for i in range(len(data_list))]
    random.shuffle(indices)
    for i in range(len(data_list)):
        data[indices[i]] = data_list[i]
        if i > 100000:
            break
    data = dict(sorted(data.items()))
    np.save(data_out, data)

def process_big_kaggle(args):
    print('----------------------------------------------')
    print('Processing dataset big_kaggle (puzzles0_kaggle)')
    print('----------------------------------------------')
    data_name = 'puzzles0_kaggle'
    data_new_name = 'big_kaggle'
    if not isdir('data/' + data_new_name):
        os.mkdir('data/' + data_new_name)
    format_conversion(data_name,data_new_name)
    assert args.solver in ['default','prolog','backtrack'] , 'choose a solver in [default, prolog, backtrack]'
    solver = args.solver
    if args.solver== 'default':
        solver = 'backtrack'
    
    dataset_generation(data_new_name,solver)
    

def process_minimal_17(args):
    print('----------------------------------------------')
    print('Processing dataset minimal_17 (puzzles2_17_clue)')
    print('----------------------------------------------')
    data_name = 'puzzles2_17_clue'
    data_new_name = "minimal_17"
    if not isdir('data/' + data_new_name):
        os.mkdir('data/' + data_new_name)
    format_conversion(data_name,data_new_name)
    assert args.solver in ['default','prolog','backtrack'] , 'choose a solver in [default, prolog, backtrack]'
    solver = args.solver
    if args.solver== 'default':
        solver = 'prolog'

    dataset_generation(data_new_name,solver)


def process_multiple_sol(args):
    print('----------------------------------------------')
    print('Processing dataset multiple_sol (puzzles7_serg_benchmark)')
    print('----------------------------------------------')
    data_name = 'puzzles7_serg_benchmark'
    data_new_name = 'multiple_sol'
    if not isdir('data/' + data_new_name):
        os.mkdir('data/' + data_new_name)
    format_conversion(data_name,data_new_name)
    assert args.solver in ['default','prolog','backtrack'] , 'choose a solver in [default, prolog, backtrack]'
    solver = args.solver
    if args.solver== 'default':
        solver = 'backtrack'

    dataset_generation(data_new_name,solver)


def statistics_datasets(dataset_name):
    print(f'---- Statistics for {dataset_name} dataset ----')
    data_in = 'data/original_data/'+ dataset_name 
    file_in = open(data_in, 'r')
    lines = file_in.readlines()
    data_list = []
    non_zero = 0
    min_nz = 81
    max_nz = 0
    count = 0
    for line in tqdm(lines):
        if '#' not in line and len(line)>80:
            count += 1
            input_line = line.replace('.','0').replace('\n','')
            input_line = np.array([int(i) for i in input_line])
            data_list.append(input_line)
            non_zero_tmp = np.count_nonzero(input_line)
            non_zero += non_zero_tmp
            if non_zero_tmp < min_nz:
                min_nz=non_zero_tmp
            if non_zero_tmp > max_nz:
                max_nz=non_zero_tmp
    file_in.close()
    non_zero /= len(data_list)
    print(f'Non zero avg for {dataset_name}: {non_zero}')
    print(f'Min: {min_nz}')
    print(f'Max: {max_nz}')
    print(f'Size: {count}')


def process_inputs(X, Y):
        is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()
        X = X.view(X.size(0), -1)
        Y = Y.view(Y.size(0), -1)
        is_input = is_input.view(is_input.size(0), -1)
        return X, Y, is_input


def statistics_satnet():
    print('---- Statistics for satnet dataset ----')
    with open('data/original_data/features.pt', 'rb') as f:
        X_in = torch.load(f)
    with open('data/original_data/labels.pt', 'rb') as f:
        Y_in = torch.load(f)
    data_s, _, _ = process_inputs(X_in,  Y_in)
    num_hints = []
    min_nz = 81
    max_nz = 0
    non_zero = 0
    for i in data_s:
        input_line = i.reshape([81,9])
        a = torch.zeros((1,81))
        input_line = torch.cat((torch.t(a),input_line),1)
        for i in range(81):
            if 1 not in input_line[i]:
                input_line[i][0]= 1.
        input_line = input_line.argmax(dim=1)
        non_zero_tmp = np.count_nonzero(input_line)
        num_hints.append(non_zero)
        if non_zero_tmp < min_nz:
                min_nz=non_zero_tmp
        if non_zero_tmp > max_nz:
            max_nz=non_zero_tmp
        non_zero += non_zero_tmp
    non_zero /= len(data_s)
    print(f'Non zero avg for satnet: {non_zero}')
    print(f'Min: {min_nz}')
    print(f'Max: {max_nz}')
    print(f'Size: {len(data_s)}')


def main_data_gen():
    parser = init_parser()
    args = parser.parse_args()

    if args.dataset == 'multiple_sol':
        process_multiple_sol(args) # multiple_sol
    elif args.dataset == 'minimal_17':
        process_minimal_17(args) # minimal_17
    elif args.dataset == 'big_kaggle':
        process_big_kaggle(args) # big_kaggle
    elif args.dataset == 'satnet_data':
        process_satnet_data(args) # satnet_data
    else:
        print(' Generating all datasets...')
        process_multiple_sol(args) # multiple_sol
        process_minimal_17(args) # minimal_17
        process_big_kaggle(args) # big_kaggle
        process_satnet_data(args) # satnet_data

if __name__=='__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
   
    main_data_gen()
    
    # statistics_datasets('puzzles0_kaggle')
    # statistics_datasets('puzzles7_serg_benchmark')
    # statistics_datasets('puzzles2_17_clue')
    # statistics_satnet()