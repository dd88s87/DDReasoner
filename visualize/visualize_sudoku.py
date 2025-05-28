import numpy as np
import matplotlib.pyplot as plt


data_name = "big_kaggle"
data_path = "data/"+data_name+"/"+data_name+".npy"
sol_path = "data/"+data_name+"/"+data_name+"_sol.npy"
datafiles = np.load(data_path, allow_pickle=True).item()
solfiles = np.load(sol_path, allow_pickle=True).item()

sudoku = datafiles[0]
# sudoku = solfiles[0]

def plot_sudoku(grid, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)

    for i in range(10):
        ax.axhline(i, color='gray', linewidth=1)
        ax.axvline(i, color='gray', linewidth=1)

    for i in range(0, 10, 3):
        ax.axhline(i, color='black', linewidth=3)
        ax.axvline(i, color='black', linewidth=3)

    for y in range(9):
        for x in range(9):
            num = grid[y, x]
            if num != 0:
                ax.text(x + 0.5, 8.5 - y, str(num),
                        va='center', ha='center', fontsize=28)
            else:
                ax.text(x + 0.5, 8.5 - y, str(num),
                        va='center', ha='center', fontsize=28)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    

# save picture
plot_sudoku(sudoku, save_path='sudoku')
