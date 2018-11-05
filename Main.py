import math
import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


class DiffEquation:
    def __init__(self, x0, y0, xn, n, f, exact_sol, formula):
        self.x0 = x0
        self.y0 = y0
        self.xn = xn
        self.n = n
        self.f = f
        self.exact_sol = exact_sol
        self.formula = formula


class Method:
    def start(self, diff_eq):
        h = (diff_eq.xn - diff_eq.x0)/diff_eq.n
        xs = list(np.linspace(diff_eq.x0, diff_eq.xn, diff_eq.n))
        ys = [diff_eq.y0]

        return h, xs, ys


class Exact(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs[1:]:
            if x != 0:
                y = diff_eq.exact_sol(x, diff_eq.x0, diff_eq.y0)
                ys.append(y)

        if 0 in xs:
            xs.remove(0)

        return xs, ys, "Exact solution"


class Euler(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs[1:]:
            if x != 0:
                y = ys[-1] + h*diff_eq.f(x, ys[-1])
                ys.append(y)

        if 0 in xs:
            xs.remove(0)

        return xs, ys, "Euler method"


class ModEuler(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs[1:]:
            if x != 0 and x + h != 0:
                k1 = diff_eq.f(x, ys[-1])
                k2 = diff_eq.f(x + h, ys[-1] + h*k1)
                y = ys[-1] + h*((k1 + k2)/2)
                ys.append(y)

        xs = list(filter(lambda x: x != 0 and x != -h, xs))

        return xs, ys, "Modified Euler method"


class RungeKutta(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs[1:]:
            if x != 0 and x + h != 0 and x + h/2 != 0:
                k1 = h*diff_eq.f(x, ys[-1])
                k2 = h*diff_eq.f(x + h/2, ys[-1] + k1/2)
                k3 = h*diff_eq.f(x + h/2, ys[-1] + k2/2)
                k4 = h*diff_eq.f(x + h, ys[-1] + k3)
                y = ys[-1] + k1/6 + k2/3 + k3/3 + k4/6
                ys.append(y)

        xs = list(filter(lambda x: x != 0 and x != -h and x != -h/2, xs))

        return xs, ys, "Runge Kutta method"


class ODESolver:
    def solve(self, diff_eq, method):
        return method.start(diff_eq)


class MainWindow:
    def create(self):
        root = tk.Tk()

        root.title("ODE Solver")
        root.resizable(width=False, height=False)
        root.iconbitmap(default="favicon.ico")

        root.geometry('1000x630')

        return root


class ControlPanel:
    def __init__(self, root, tabs, diff_eq, solver, method):
        self.root = root
        self.tabs = tabs
        self.diff_eq = diff_eq
        self.solver = solver
        self.method = method

    def create(self):
        control_panel = ttk.Frame(self.root, padding="15")
        control_panel.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        ttk.Label(control_panel, text="x0")
        entry_x0 = ttk.Entry(control_panel)
        entry_x0.insert(tk.END, self.diff_eq.x0)

        ttk.Label(control_panel, text="y0")
        entry_y0 = ttk.Entry(control_panel)
        entry_y0.insert(tk.END, self.diff_eq.y0)

        ttk.Label(control_panel, text="xn")
        entry_xn = ttk.Entry(control_panel)
        entry_xn.insert(tk.END, self.diff_eq.xn)

        ttk.Label(control_panel, text="n")
        entry_n = ttk.Entry(control_panel)
        entry_n.insert(tk.END, self.diff_eq.n)

        btn = ttk.Button(control_panel, text="Plot",
                         command=lambda e_x0=entry_x0, e_y0=entry_y0, e_xn=entry_xn, e_n=entry_n:
                         self.update(int(e_x0.get()), int(e_y0.get()), int(e_xn.get()), int(e_n.get())))

        btn.grid(column=0, columnspan=2, row=5, sticky=(tk.W, tk.E))

        i = 0
        for child in control_panel.winfo_children():
            child.grid_configure(padx=5, pady=5)

            if isinstance(child, ttk.Label):
                child.grid(column=0, row=i, sticky=(tk.W, tk.E))

            if isinstance(child, ttk.Entry):
                child.config(justify=tk.CENTER)
                child.grid(column=1, row=i, sticky=(tk.W, tk.E))
                i = i + 1

        self.update(int(entry_x0.get()), int(entry_y0.get()), int(entry_xn.get()), int(entry_n.get()))

    def update(self, x0, y0, xn, n):
        if x0 == 0:
            msg.showinfo("Error", "Division by 0!\nThe value x0 = 0 isn't permitted.")
        else:
            self.diff_eq.x0 = x0
            self.diff_eq.y0 = y0
            self.diff_eq.xn = xn
            self.diff_eq.n = n

            plots = [
                self.solver.solve(self.diff_eq, self.method["Exact"]),
                self.solver.solve(self.diff_eq, self.method["Euler"]),
                self.solver.solve(self.diff_eq, self.method["ModEuler"]),
                self.solver.solve(self.diff_eq, self.method["RungeKutta"])
            ]

            for tab in self.tabs:
                tab.update(plots)

    def plot(self, plots):
        plt.figure(figsize=(6, 7))
        plt.suptitle(self.diff_eq.formula + "\nx0 = " + str(self.diff_eq.x0) + ", xn = " + str(self.diff_eq.xn)
                     + ", y0 = " + str(self.diff_eq.y0) + ", n = " + str(self.diff_eq.n))

        plt.subplot(211)
        plt.title("Solutions")
        plt.xlabel("x")
        plt.ylabel("y")

        for plot in plots:
            plt.plot(plot[0], plot[1], label=plot[2])
        plt.subplots_adjust(hspace=0.5)
        plt.legend()

        plt.subplot(212)
        plt.title("Errors")
        plt.xlabel("x")
        plt.ylabel("y")

        if len(plots) > 1 and plots[0][2] == "Exact solution":
            for plot in plots[1:]:
                plt.plot(plot[0], abs(np.array(plot[1]) - np.array(plots[0][1])), label="Error of " + plot[2])
            plt.legend()

        plt.show()


class GraphArea:
    def __init__(self, root):
        self.root = root

    def create(self):
        graph_area = ttk.Notebook(self.root, padding="15", width=750, height=550)
        graph_area.pack(fill='both', expand='yes')
        graph_area.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        return graph_area


class Tab:
    def __init__(self, graph_area):
        tab = tk.Frame(graph_area, width=750, height=400)
        fig = Figure(figsize=(5, 5), dpi=100)

        self.graph = fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.item = tab
        self.canvas = canvas
        self.toolbar = toolbar

    def update(self, plots):
        self.graph.clear()

        for plot in plots:
            self.graph.plot(plot[0], plot[1], label=plot[2])
            self.graph.set_xlabel("x")
            self.graph.set_ylabel("y")
        self.canvas.draw()
        self.toolbar.update()


class ODESolverApp:
    def __init__(self, diff_eq, solver, method):
        root = MainWindow().create()
        graph_area = GraphArea(root).create()

        tabs = [
            Tab(graph_area),
            Tab(graph_area),
            Tab(graph_area)
        ]

        graph_area.add(tabs[0].item, text="Solution")
        graph_area.add(tabs[1].item, text="Local error")
        graph_area.add(tabs[2].item, text="Total approximation error")

        ControlPanel(root, tabs, diff_eq, solver, method).create()
        root.mainloop()


def ode_func(x, y):
    return 3*x*math.e**x - y*(1 - 1/x)


def ode_exact_sol(x, x0, y0):
    c = y0*(math.e**x0/x0) - 3/2*math.e**(2*x0)
    return 3/2*x*math.e**x + c*(x/math.e**x)


ode = DiffEquation(1, 0, 5, 5, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
ode_solver = ODESolver()
methods_list = {"Exact": Exact(), "Euler": Euler(), "ModEuler": ModEuler(), "RungeKutta": RungeKutta()}

ODESolverApp(ode, ode_solver, methods_list)
