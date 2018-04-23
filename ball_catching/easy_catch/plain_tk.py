#!/usr/bin/env python

import Tkinter as Tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class Model:
    def __init__(self):
        self.xpoint = 200
        self.ypoint = 200
        self.res = None

    def calculate(self):
        x, y = np.meshgrid(
            np.linspace(-5, 5, self.xpoint),
            np.linspace(-5, 5, self.ypoint)
        )
        z = np.cos(x**2 * y**3)
        self.res = {'x': x, 'y': y, 'z': z}


class View:
    def __init__(self, master):
        self.frame = Tk.Frame(master)
        self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

        self.fig = Figure(figsize=(7.5, 4), dpi=80)
        self.ax0 = self.fig.add_axes(
            (0.05, .05, .90, .90), axisbg=(.75, .75, .75), frameon=False
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.show()

        self.side_panel = SidePanel(master)


class Controller:
    def __init__(self):
        self.root = Tk.Tk()
        self.model = Model()
        self.view = View(self.root)

        self.view.side_panel.plot_button.bind("<Button>", self.plot)
        self.view.side_panel.clear_button.bind("<Button>", self.clear)

    def run(self):
        self.root.title("Tkinter MVC example")
        # self.root.deiconify()
        self.root.mainloop()

    def plot(self, event):
        self.model.calculate()
        self.view.ax0.clear()
        self.view.ax0.contourf(
            self.model.res['x'], self.model.res['y'], self.model.res['z']
        )
        self.view.fig.canvas.draw()

    def clear(self, event):
        self.view.ax0.clear()
        self.view.fig.canvas.draw()


class SidePanel:
    def __init__(self, master):
        self.frame = Tk.Frame(master)
        self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

        self.plot_button = Tk.Button(self.frame, text="Plot")
        self.plot_button.pack(side="top", fill=Tk.BOTH)
        self.clear_button = Tk.Button(self.frame, text="Clear")
        self.clear_button.pack(side="top", fill=Tk.BOTH)


if __name__ == '__main__':
    c = Controller()
    c.run()
