import math
import shutil

import numpy as np
from matplotlib import pyplot as plt, animation


def lim(a, margin):
    """
    Adjusts the display range of data to provide better visualization.
    
    This method calculates a new minimum and maximum value based on the input array
    and a given margin. It expands the range to ensure important features are visible
    without clipping, improving data interpretability.
    
    Args:
        a (numpy.ndarray): The input array of numerical data.
        margin (float): The margin as a percentage of the original range.
    
    Returns:
        tuple: A tuple containing the new minimum and maximum values (min_val, max_val).
    """
    min_ = np.min(a)
    max_ = np.max(a)
    m = margin * (max_ - min_)
    return (min_ - m, max_ + m)


class Animation:
    """
    A class for generating and saving animations from a series of lines.
    
        Attributes:
            step: The interval between points.
            interval: The time interval between frames.
            max_points: The maximum number of points to store.
            i: The current index of the point being generated.
            x: A list to store the x-coordinates of the points.
            y: A list to store the y-coordinates of the points.
    """

    def __init__(self, num, fps=30, length=10.0, max_points=512):
        """
        Initializes a new instance of the class.
        
        Args:
            num (int): The total number of points to generate for the animation.
            fps (int, optional): Frames per second for the animation. Defaults to 30.
            length (float, optional): The length of the animation in seconds. Defaults to 10.0.
            max_points (int, optional): The maximum number of points to store. Defaults to 512.
        
        Initializes the following class fields:
            step (int): The interval between points, calculated to ensure a smooth animation based on the desired length and frame rate.
            interval (float): The time interval between frames, adjusted to maintain the specified frame rate while accommodating the generated points.
            max_points (int): The maximum number of points that can be stored.
            i (int): The current index of the point being generated.
            x (list): A list to store the x-coordinates of the points.
            y (list): A list to store the y-coordinates of the points.
        
        Returns:
            None
        """
        min_interval = 1 / fps
        max_frames = math.ceil(length / min_interval)
        self.step = math.ceil(num / max_frames)

        num_frames = len(range(0, num, self.step))
        self.interval = max(length / num_frames, min_interval)

        self.max_points = max_points
        self.i = 0
        self.x = []
        self.y = []

    def update(self, lines):
        """
        Updates the internal data with new lines, sampling points if necessary.
        
        Args:
            self: The instance of the class.
            lines: A list of tuples, where each tuple contains two lists representing x and y coordinates.
        
        This method processes incoming coordinate data by selectively adding sampled points
        to maintain a manageable data size while preserving the overall trend. It calculates
        a sampling interval based on the desired maximum number of points and the length
        of the input data, ensuring that the data remains representative without becoming
        excessively large.
        
        Args:
            self: The instance of the class.
            lines (list): A list of tuples, where each tuple contains two lists representing x and y coordinates.
        
        Returns:
            None
        """
        if self.i % self.step == 0:
            xs = []
            ys = []
            for (x, y) in lines:
                step = math.ceil(len(x) / self.max_points)
                xs.append(x[::step].copy())
                ys.append(y[::step].copy())
            self.x.append(xs)
            self.y.append(ys)
        self.i += 1

    def save(self, out_name, x_margin=0.1, y_margin=0.1):
        """
        Saves the animation to an HTML file.
        
        This method generates an animation from the stored data, configures the plot axes,
        and saves the animation as an HTML file. It prepares the visualization by setting
        axis limits based on the data and specified margins, then creates and saves the animation.
        Existing files with the same name are removed to prevent conflicts.
        
        Args:
            self: The instance of the Animation class.
            out_name (str): The base name for the output HTML file (without extension).
            x_margin (float, optional): The margin to add to the x-axis limits. Defaults to 0.1.
            y_margin (float, optional): The margin to add to the y-axis limits. Defaults to 0.1.
        
        Returns:
            None
        """
        (fig, axes) = plt.subplots()
        plots = [axes.plot([], [])[0] for _ in self.x[0]]
        axes.set_xlim(*lim(self.x, x_margin))
        axes.set_ylim(*lim(self.y, y_margin))

        def update(frame):
            for (plot, x, y) in zip(plots, self.x[frame], self.y[frame]):
                plot.set_data(x, y)
            return plots

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.x),
                                      interval=self.interval * 1000)
        shutil.rmtree(f'{out_name}_frames', ignore_errors=True)
        ani.save(filename=f'{out_name}.html', writer="html")
