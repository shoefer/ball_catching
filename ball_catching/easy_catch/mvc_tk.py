#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, cos, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


root = Tk.Tk()
root.wm_title("Embedding in TK")


f = Figure()
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)
c = cos(2*pi*t)

# A tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

# Plot functions
def plot_sin():
    a.clear()
    a.plot(t, s)
    a.set_title('Sin')
    a.set_xlabel('X axis label')
    a.set_ylabel('Y label')
    canvas.show()

def plot_cos():
    a.clear()
    a.plot(t, c)
    a.set_title('Cos')
    a.set_xlabel('X axis label')
    a.set_ylabel('Y label')
    canvas.show()

# Buttons
quit_button = Tk.Button(master=root, text='Quit', command=sys.exit)
quit_button.pack(side=Tk.LEFT)

sin_button = Tk.Button(master=root, text='sin', command=plot_sin)
sin_button.pack(side=Tk.LEFT)

cos_button = Tk.Button(master=root, text='cos', command=plot_cos)
cos_button.pack(side=Tk.LEFT)

Tk.mainloop()