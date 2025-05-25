import numpy as np
import matplotlib.pyplot as plt

path = ""  # set your path
maze_solved = np.load(path)
map = maze_solved[:,:,0]
goal = maze_solved[:,:,1]   
path = maze_solved[:,:,2]


def plot_maze_from_pixel_grid(pixel_grid, save_path=None):
    """
    Plot the maze based on the given pixel grid and save the image.
    - WALL = 1 -> black
    - START = 0 -> green
    - END = 1 -> purple
    - PATH_POINT = 1 -> blue
    - Other regions -> white
    
    Args:
        pixel_grid (numpy.ndarray): The pixel grid representing the maze (shape: [H, W, 3]).
        save_path (str, optional): The path to save the plotted image. If None, the plot will be displayed instead.
    """
    # Create an empty RGB grid to store the color information (H, W, 3)
    H, W, _ = pixel_grid.shape

    # Create an empty RGB grid to store the color information (H, W, 3)
    maze_rgb = np.ones((H, W, 3), dtype=np.float32)  # Initialize with white color

    # WALL = 1 -> black
    maze_rgb[pixel_grid[:, :, 0] == -1] = [0, 0, 0]  # Set wall to black

    # PATH_POINT = 1 -> blue (path[2] = blue)
    maze_rgb[pixel_grid[:, :, 2] == 1] = [0, 0, 1] # Set path to blue

    # START = 0 -> green (path[0,1] = green)
    maze_rgb[pixel_grid[:, :, 1] == 0] = [0, 1, 0]  # Set start point to green

    # END = 1 -> purple (path[0,2] = purple)
    maze_rgb[pixel_grid[:, :, 1] == 1] = [1, 0, 1]  # Set end point to purple

    # Plotting the maze
    plt.figure(figsize=(H / 10, W / 10))
    plt.imshow(maze_rgb)
    plt.axis('off')  # Hide axes

    # Saving or showing the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Maze saved to {save_path}")
    else:
        plt.show()  # Show the maze if no save path is provided

    plt.close()
    
plot_maze_from_pixel_grid(maze_solved, "maze")