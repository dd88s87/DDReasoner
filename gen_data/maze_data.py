from zanj import ZANJ  # saving/loading data
import argparse
import os
import sys
import numpy as np

# maze_dataset imports
from maze_dataset import SolvedMaze, MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

# for saving/loading things
LOCAL_DATA_PATH: str = "data/maze_dataset/"
zanj: ZANJ = ZANJ(external_list_threshold=256)

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))

sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser(description="Generate a dataset of mazes")
parser.add_argument("--dataset_name", type=str, default="Maze-10", help="Name of the dataset")
parser.add_argument("--grid_n", type=int, default=10, help="Number of rows/columns in the lattice")
parser.add_argument("--n_mazes", type=int, default=30000, help="Number of mazes to generate")
parser.add_argument("--maze_ctor", type=str, default="gen_dfs", help="Algorithm to generate the maze")
parser.add_argument("--do_download", type=str2bool, default=False, help="Download the dataset")
parser.add_argument("--load_local", type=str2bool, default=False, help="Load the dataset locally")
parser.add_argument("--do_generate", type=str2bool, default=True, help="Generate the dataset")
parser.add_argument("--save_local", type=str2bool, default=False, help="Save the dataset locally")
parser.add_argument("--local_base_path", type=str, default="data/maze", help="Base path for local storage")
parser.add_argument("--verbose", type=str2bool, default=True, help="Print information about the dataset")
parser.add_argument("--gen_parallel", type=str2bool, default=True, help="Generate the mazes in parallel")
parser.add_argument("--min_length", type=int, default=5, help="Minimum length of the maze")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the maze")

args = parser.parse_args()
args.local_base_path = args.local_base_path + "/" + args.dataset_name+f'grid_n-{args.grid_n}_n_mazes-{args.n_mazes}_min_length-{args.min_length}_max_length-{args.max_length}'
    
cfg: MazeDatasetConfig = MazeDatasetConfig(
    name=args.dataset_name,  # name of the dataset
    grid_n=args.grid_n,  # number of rows/columns in the lattice
    n_mazes=args.n_mazes,  # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_dfs,  # algorithm to generate the maze
)

dataset: MazeDataset = MazeDataset.from_config(
    cfg,
    do_download=args.do_download,
    load_local=args.load_local,
    do_generate=args.do_generate,
    save_local=args.save_local,
    local_base_path=args.local_base_path,
    verbose=args.verbose,
    zanj=zanj,
    gen_parallel=args.gen_parallel,
)

dataset_filtered: MazeDataset = dataset.filter_by.path_length(min_length=args.min_length)

if __name__=='__main__':
    WALL = 1
    REE = -1
    START = 0
    END = 1
    PATH_POINT = 1
    args.local_base_path = args.local_base_path +'N-'+str(len(dataset_filtered))
    if not os.path.exists(args.local_base_path):
        os.makedirs(args.local_base_path)

    for i in range(len(dataset_filtered)):
        data_i = dataset_filtered[i]
        pixel_grid_bw = data_i._as_pixels_bw()
        pixel_grid = np.full(
            (*pixel_grid_bw.shape, 3), -1, dtype=np.int8
        ) # set all to -1 [H,W,3]
        # set map
        pixel_grid[pixel_grid_bw == True,0] = WALL

        # Set goal
        pixel_grid[data_i.start_pos[0] * 2 + 1, data_i.start_pos[1] * 2 + 1,1] = START
        pixel_grid[data_i.end_pos[0] * 2 + 1, data_i.end_pos[1] * 2 + 1,1] = END

        # Set path
        for coord in data_i.solution:
            pixel_grid[coord[0] * 2 + 1, coord[1] * 2 + 1,2] = PATH_POINT
        ## set pixels between coords
        for index, coord in enumerate(data_i.solution[:-1]):
            next_coord = data_i.solution[index + 1]
            # check they are adjacent using norm
            assert (
                np.linalg.norm(np.array(coord) - np.array(next_coord)) == 1
            ), f"Coords {coord} and {next_coord} are not adjacent"
            # set pixel between them
            pixel_grid[
                coord[0] * 2 + 1 + next_coord[0] - coord[0],
                coord[1] * 2 + 1 + next_coord[1] - coord[1],2
            ] = PATH_POINT
        np.save(f"{args.local_base_path}/maze_solved-{i}.npy", pixel_grid)

    print(f"Done! {len(dataset_filtered)} datapoints saved to {args.local_base_path}")