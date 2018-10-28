import math
import numpy
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk


class DiffEquation:
    def __init__(self, x0, y0, xn, h, f, exact_sol, formula):
        self.x0 = x0
        self.y0 = y0
        self.xn = xn
        self.h = h
        self.f = f
        self.exact_sol = exact_sol
        self.formula = formula


class Method:
    def start(self, diff_eq):
        pass


class Exact(Method):
    def start(self, diff_eq):
        xs = numpy.arange(diff_eq.x0, diff_eq.xn + diff_eq.h, diff_eq.h)
        ys = [diff_eq.y0]

        for x in xs[1:]:
            y = diff_eq.exact_sol(x)
            ys.append(y)

        return xs, ys, "Exact solution"


class Euler(Method):
    def start(self, diff_eq):
        xs = numpy.arange(diff_eq.x0, diff_eq.xn + diff_eq.h, diff_eq.h)
        ys = [diff_eq.y0]

        for x in xs[1:]:
            y = ys[-1] + diff_eq.h*diff_eq.f(x, ys[-1])
            ys.append(y)

        return xs, ys, "Euler method"


class ModEuler(Method):
    def start(self, diff_eq):
        xs = numpy.arange(diff_eq.x0, diff_eq.xn + diff_eq.h, diff_eq.h)
        ys = [diff_eq.y0]

        for x in xs[1:]:
            k1 = diff_eq.f(x, ys[-1])
            k2 = diff_eq.f(x + diff_eq.h, ys[-1] + diff_eq.h*k1)
            y = ys[-1] + diff_eq.h*((k1 + k2)/2)
            ys.append(y)

        return xs, ys, "Modified Euler method"


class RungeKutta(Method):
    def start(self, diff_eq):
        xs = numpy.arange(diff_eq.x0, diff_eq.xn + diff_eq.h, diff_eq.h)
        ys = [diff_eq.y0]

        for x in xs[1:]:
            k1 = diff_eq.h*diff_eq.f(x, ys[-1])
            k2 = diff_eq.h*diff_eq.f(x + diff_eq.h/2, ys[-1] + k1/2)
            k3 = diff_eq.h*diff_eq.f(x + diff_eq.h/2, ys[-1] + k2/2)
            k4 = diff_eq.h*diff_eq.f(x + diff_eq.h, ys[-1] + k3)
            ys.append(ys[-1] + k1/6 + k2/3 + k3/3 + k4/6)

        return xs, ys, "Runge Kutta method"


class ODESolver:
    def solve(self, diff_eq, method):
        return method.start(diff_eq)


def plot(m0, m1, m2, m3, diff_eq):
    plt.plot(m0[0], m0[1], label=m0[2])
    plt.plot(m1[0], m1[1], label=m1[2])
    plt.plot(m2[0], m2[1], label=m2[2])
    plt.plot(m3[0], m3[1], label=m3[2])

    plt.title(diff_eq.formula + ", h = " + str(diff_eq.h))
    plt.legend()
    plt.show()


def ode_func(x, y):
    return 3*x*math.e**x - y*(1 - 1/x)


def ode_exact_sol(x):
    return 3/2*x*(math.e**x - math.e**(2 - x))


diff_eq = DiffEquation(1, 0, 5, 0.1, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
solver = ODESolver()

m0 = solver.solve(diff_eq, Exact())
m1 = solver.solve(diff_eq, Euler())
m2 = solver.solve(diff_eq, ModEuler())
m3 = solver.solve(diff_eq, RungeKutta())

# plot(m0, m1, m2, m3, diff_eq)


def plot2(*args):
    try:
        x0 = entry_x0.get()
        y0 = entry_y0.get()
        xn = entry_xn.get()
        h = entry_h.get()

        print(x0, y0, xn, h)
    except ValueError:
        pass


root = Tk()
root.title("ODE Solver")
root.resizable(width=False, height=False)
root.geometry('1000x600')

control_panel = ttk.Frame(root, padding="15")
control_panel.grid(column=0, row=0, sticky=(N, W, E, S))

ttk.Label(control_panel, text="x0")
entry_x0 = ttk.Entry(control_panel)
entry_x0.insert(END, '1')

ttk.Label(control_panel, text="y0")
entry_y0 = ttk.Entry(control_panel)
entry_y0.insert(END, '0')

ttk.Label(control_panel, text="xn")
entry_xn = ttk.Entry(control_panel)
entry_xn.insert(END, '5')

ttk.Label(control_panel, text="h")
entry_h = ttk.Entry(control_panel)
entry_h.insert(END, '0.1')

ttk.Button(control_panel, text="Plot", command=plot2).grid(column=0, columnspan=2, row=5, sticky=(W, E))

i = 0
for child in control_panel.winfo_children():
    child.grid_configure(padx=5, pady=5)
    if isinstance(child, ttk.Label):
        child.grid(column=0, row=i, sticky=(W, E))
    if isinstance(child, ttk.Entry):
        child.config(justify=CENTER)
        child.grid(column=1, row=i, sticky=(W, E))
        i = i + 1

graph_area = ttk.Frame(root, padding="15", width=800, height=600)
graph_area.grid(column=1, row=0, sticky=(N, W, E, S))

nb = ttk.Notebook(graph_area)
nb.pack(fill='both', expand='yes')

page_1 = Frame(graph_area, width=750, height=520)
# -----------
f = Figure(figsize=(5, 5), dpi=100)
a = f.add_subplot(111)
a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])

canvas = FigureCanvasTkAgg(f, page_1)
canvas.draw()
canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

toolbar = NavigationToolbar2Tk(canvas, page_1)
toolbar.update()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
# ------------
page_2 = Frame(graph_area, width=750, height=520)
nb.add(page_1, text='Solutions')
nb.add(page_2, text='Errors')

root.bind('<Return>', plot2)    # action on pressing 'Enter'

root.mainloop()
