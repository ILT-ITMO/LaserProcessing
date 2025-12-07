import math
import shutil

import numpy as np
from matplotlib import pyplot as plt, animation


def lim(a, margin):
    min_ = np.min(a)
    max_ = np.max(a)
    m = margin * (max_ - min_)
    return (min_ - m, max_ + m)


class Animation:
    def __init__(self, num, fps=30, length=10.0, max_points=512):
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
