# Data Generation

## Download the data
1- Sudoku:
* [dowload Tdoku datasets from here](https://github.com/t-dillon/tdoku/blob/master/data.zip)
unzip the data (9 files) and place `puzzles0_kaggle`, `puzzles2_17_clue`, and `puzzles7_serg_benchmark` in `gen_data/original_data`
* [download SatNet dataset from here](https://powei.tw/sudoku.zip)
unzip the data (4 files) and place `features.pt` and `labels.pt` in `gen_data/original_data`

Install Prolog (instructions [here](https://www.swi-prolog.org/Download.html)).
Run `python gen_data/sudoku_data.py --solver default` to generate the data. With problems installing prolog use `--solver backtrack` (it might take longer to run). 

2- Maze:
Install maze-dataset (instructions [here](https://github.com/understanding-search/maze-dataset)).
Run `python gen_data/maze_data.py`

3- Warcraft Shortest Path
Download ([here] https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S)

4- Simple Path Prediction & Preference Learning
The data is placed in `gen_data/data/grid` and `gen_data/data/sushi`

## Set the data
After all data has been generated, move the folder `data` to the parent directory (at the same level as `gen_data`).