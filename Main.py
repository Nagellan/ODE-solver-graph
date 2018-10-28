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


class ODESolverApp:
    def __init__(self, diff_eq, m0, m1, m2, m3):
        root = Tk()
        root.title("ODE Solver")
        root.resizable(width=False, height=False)
        root.geometry('1000x600')
        # root.iconbitmap(default="graph-img.gif")

        self.root = root

        self.draw_control_panel(diff_eq)
        self.draw_graph_area(m0, m1, m2, m3)

        root.mainloop()

    def update(*args):
        try:
            pass
        except ValueError:
            pass

    def draw_control_panel(self, diff_eq):
        control_panel = ttk.Frame(self.root, padding="15")
        control_panel.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(control_panel, text="x0")
        entry_x0 = ttk.Entry(control_panel)
        entry_x0.insert(END, diff_eq.x0)

        ttk.Label(control_panel, text="y0")
        entry_y0 = ttk.Entry(control_panel)
        entry_y0.insert(END, diff_eq.y0)

        ttk.Label(control_panel, text="xn")
        entry_xn = ttk.Entry(control_panel)
        entry_xn.insert(END, diff_eq.xn)

        ttk.Label(control_panel, text="h")
        entry_h = ttk.Entry(control_panel)
        entry_h.insert(END, diff_eq.h)

        ttk.Button(control_panel, text="Plot", command=self.update).grid(column=0, columnspan=2, row=5, sticky=(W, E))

        i = 0
        for child in control_panel.winfo_children():
            child.grid_configure(padx=5, pady=5)
            if isinstance(child, ttk.Label):
                child.grid(column=0, row=i, sticky=(W, E))
            if isinstance(child, ttk.Entry):
                child.config(justify=CENTER)
                child.grid(column=1, row=i, sticky=(W, E))
                i = i + 1

        self.root.bind('<Return>', self.update)  # action on pressing 'Enter'

    def draw_graph_area(self, m0, m1, m2, m3):
        graph_area = ttk.Frame(self.root, padding="15", width=800, height=600)
        graph_area.grid(column=1, row=0, sticky=(N, W, E, S))

        nb = ttk.Notebook(graph_area)
        nb.pack(fill='both', expand='yes')

        tab_1 = self.create_tab_1(graph_area, m0, m1, m2, m3)
        tab_2 = self.create_tab_2(graph_area, m0, m1, m2, m3)

        nb.add(tab_1, text='Solutions')
        nb.add(tab_2, text='Errors')

    def create_tab_1(self, graph_area, m0, m1, m2, m3):
        tab = Frame(graph_area, width=750, height=520)

        fig = Figure(figsize=(5, 5), dpi=100)
        graph = fig.add_subplot(111)
        graph.plot(m0[0], m0[1])
        graph.plot(m1[0], m1[1])
        graph.plot(m2[0], m2[1])
        graph.plot(m3[0], m3[1])

        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        return tab

    def create_tab_2(self, graph_area, m0, m1, m2, m3):
        tab = Frame(graph_area, width=750, height=520)
        fig = Figure(figsize=(5, 5), dpi=100)
        graph = fig.add_subplot(111)
        graph.plot(m1[0], numpy.array(m1[1] - numpy.array(m0[1])))
        graph.plot(m2[0], numpy.array(m2[1] - numpy.array(m0[1])))
        graph.plot(m3[0], numpy.array(m3[1] - numpy.array(m0[1])))

        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        return tab


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


diff_eq = DiffEquation(1, 0, 5, 1, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
solver = ODESolver()

m0 = solver.solve(diff_eq, Exact())
m1 = solver.solve(diff_eq, Euler())
m2 = solver.solve(diff_eq, ModEuler())
m3 = solver.solve(diff_eq, RungeKutta())

# plot(m0, m1, m2, m3, diff_eq)

ODESolverApp(diff_eq, m0, m1, m2, m3)
