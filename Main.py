import math
import sys
import os
import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


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
        h = (diff_eq.xn - diff_eq.x0)/(diff_eq.n - 1)
        xs = list(np.linspace(diff_eq.x0, diff_eq.xn, diff_eq.n))
        ys = [diff_eq.y0]

        return h, xs, ys


class Exact(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs[1:]:
            if x != 0 and x + h != 0 and x + h/2 != 0:
                y = diff_eq.exact_sol(x, diff_eq.x0, diff_eq.y0)
                ys.append(y)

        xs = list(filter(lambda x: x != 0 and x != -h and x != -h/2, xs))

        return xs, ys, "Analytical method"


class Euler(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs:
            if x != 0 and x + h != 0 and x + h/2 != 0:
                y = ys[-1] + h*diff_eq.f(x, ys[-1])
                ys.append(y)

        xs = list(filter(lambda x: x != 0 and x != -h and x != -h/2, xs))

        return xs, ys[:-1], "Euler method"


class ModEuler(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs:
            if x != 0 and x + h != 0 and x + h/2 != 0:
                k1 = diff_eq.f(x, ys[-1])
                k2 = diff_eq.f(x + h, ys[-1] + h*k1)
                y = ys[-1] + h*((k1 + k2)/2)
                ys.append(y)

        xs = list(filter(lambda x: x != 0 and x != -h and x != -h/2, xs))

        return xs, ys[:-1], "Modified Euler method"


class RungeKutta(Method):
    def start(self, diff_eq):
        h, xs, ys = super().start(diff_eq)

        for x in xs:
            if x != 0 and x + h != 0 and x + h/2 != 0:
                y = ys[-1]
                k1 = h*diff_eq.f(x, y)
                k2 = h*diff_eq.f(x + h/2, y + k1/2)
                k3 = h*diff_eq.f(x + h/2, y + k2/2)
                k4 = h*diff_eq.f(x + h, y + k3)
                y += (k1 + 2*k2 + 2*k3 + k4)/6
                ys.append(y)

        xs = list(filter(lambda x: x != 0 and x != -h and x != -h/2, xs))

        return xs, ys[:-1], "Runge Kutta method"


class ODESolver:
    def solve(self, diff_eq, method):
        return method.start(diff_eq)


class MainWindow:
    def __init__(self):
        root = tk.Tk()
        root.title("ODE Solver")
        root.resizable(width=False, height=False)
        root.iconbitmap(default=resource_path("favicon.ico"))
        root.geometry('1000x630')

        self.item = root


class ControlPanel:
    def __init__(self, root, tabs, diff_eq, solver, methods):
        self.tabs = tabs
        self.diff_eq = diff_eq
        self.solver = solver
        self.methods = methods

        control_panel = ttk.Frame(root, padding="15")
        control_panel.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        l_title = ttk.Label(control_panel, text=diff_eq.formula, font=("Arial", 14))
        l_title.grid(column=0, columnspan=2, row=0, sticky=(tk.W, tk.E), pady="20")

        ttk.Label(control_panel, text="x0")
        entry_x0 = ttk.Entry(control_panel)
        entry_x0.insert(tk.END, diff_eq.x0)

        ttk.Label(control_panel, text="y0")
        entry_y0 = ttk.Entry(control_panel)
        entry_y0.insert(tk.END, diff_eq.y0)

        ttk.Label(control_panel, text="xn")
        entry_xn = ttk.Entry(control_panel)
        entry_xn.insert(tk.END, diff_eq.xn)

        ttk.Label(control_panel, text="N")
        entry_n = ttk.Entry(control_panel)
        entry_n.insert(tk.END, diff_eq.n)

        sep = ttk.Separator(control_panel, orient=tk.HORIZONTAL)
        sep.grid(column=0, columnspan=2, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        ttk.Label(control_panel, text="n0")
        entry_n0 = ttk.Entry(control_panel)
        entry_n0.insert(tk.END, 2)

        ttk.Label(control_panel, text="n")
        entry_nn = ttk.Entry(control_panel)
        entry_nn.insert(tk.END, 10)

        i = 1
        for child in control_panel.winfo_children()[1:]:
            if i == 1:
                i = i + 1
            child.grid_configure(padx=5, pady=5)

            if isinstance(child, ttk.Label):
                child.grid(column=0, row=i, sticky=(tk.W, tk.E))

            if isinstance(child, ttk.Entry):
                child.config(justify=tk.CENTER)
                child.grid(column=1, row=i, sticky=(tk.W, tk.E))
                i = i + 1

        btn = ttk.Button(control_panel, text="Plot", padding="10",
                         command=lambda x0=entry_x0, y0=entry_y0, xn=entry_xn, n=entry_n, n0=entry_n0, nn=entry_nn:
                         self.update(int(x0.get()), int(y0.get()), int(xn.get()),
                                     int(n.get()), int(n0.get()), int(nn.get())))

        btn.grid(column=0, columnspan=2, row=i+1, sticky=(tk.W, tk.E), pady="25")

        self.update(int(entry_x0.get()), int(entry_y0.get()), int(entry_xn.get()),
                    int(entry_n.get()), int(entry_n0.get()), int(entry_nn.get()))

    def update(self, x0, y0, xn, n, n0, nn):
        def no_errors(_x0, _n0):
            message = ""
            if _x0 == 0:
                message += "Division by 0!\nThe value x0 = 0 isn't permitted.\n\n"
            if _n0 < 2:
                message += "Division by 0!\nThe value of n0 must be greater than or equal to 2."
            if len(message) > 0:
                msg.showinfo("Error(s)!", message)
                return 0
            return 1

        if no_errors(x0, n0):
            self.diff_eq.x0 = x0
            self.diff_eq.y0 = y0
            self.diff_eq.xn = xn
            self.diff_eq.n = n

            self.tabs[0].get_solutions(self.diff_eq, self.solver, self.methods)
            plots = self.tabs[0].plt
            self.tabs[0].update()

            self.tabs[1].get_local_errors(plots)
            self.tabs[1].update()

            self.tabs[2].get_global_errors(self.diff_eq, self.solver, self.methods, n0, nn)
            self.tabs[2].update()

            self.diff_eq.n = n


class GraphArea:
    def __init__(self, root):
        graph_area = ttk.Notebook(root, padding="15", width=750, height=550)
        graph_area.pack(fill='both', expand='yes')
        graph_area.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.item = graph_area


class Tab:
    title = "Empty tab"
    plt = []

    def __init__(self, graph_area, pop_up):
        tab = tk.Frame(graph_area, width=750, height=400)
        fig = Figure(figsize=(5, 5), dpi=100)

        graph = fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btn = ttk.Button(toolbar, text="Open in a new window", padding="5",
                         command=lambda this_tab=self, title=self.title: pop_up.create(this_tab.plt, title))
        btn.pack(side=tk.LEFT, padx="10", pady="5")

        self.item = tab
        self.graph = graph
        self.canvas = canvas
        self.toolbar = toolbar

    def update(self):
        self.graph.clear()

        for plot in self.plt:
            self.graph.plot(plot[0], plot[1], label=self.title + " of " + plot[2])
            self.graph.set_xlabel("x")
            self.graph.set_ylabel("y")

        self.graph.legend()
        self.canvas.draw()
        self.toolbar.update()


class TabSolution(Tab):
    title = "Solution"

    def get_solutions(self, diff_eq, solver, methods):
        plots = []
        for mtd in methods:
            plots.append(solver.solve(diff_eq, mtd))
        self.plt = plots


class TabLocalError(Tab):
    title = "Local error"

    def get_local_errors(self, plots):
        local_errors = []
        for plot in plots[1:]:
            local_errors.append((plot[0], list(abs(np.array(plot[1]) - np.array(plots[0][1]))), plot[2]))
        self.plt = local_errors


class TabTotalError(Tab):
    title = "Total approximation error"

    def get_global_errors(self, diff_eq, solver, methods, n0, nn):
        global_errors = []
        ns = np.linspace(n0, nn, nn - 1)
        for mtd in methods[1:]:
            errors = []
            plot = []
            for _n in ns:
                diff_eq.n = _n
                plot = solver.solve(diff_eq, mtd)
                exact = solver.solve(diff_eq, methods[0])
                errors.append(max(list(abs(np.array(plot[1]) - np.array(exact[1])))))
            global_errors.append((ns, errors, plot[2]))
        self.plt = global_errors


class PopUpWindow:
    def __init__(self, diff_eq):
        self.diff_eq = diff_eq

    def create(self, plots, title):
        graph_data = "x0 = " + str(self.diff_eq.x0) + ", xn = " + str(self.diff_eq.xn) + ", y0 = " \
                     + str(self.diff_eq.y0) + ", N = " + str(self.diff_eq.n)
        plt.figure(figsize=(7, 6))
        plt.suptitle(title + "\n" + self.diff_eq.formula + "\n" + graph_data)
        this_manager = plt.get_current_fig_manager()
        this_manager.window.wm_iconbitmap(resource_path("favicon.ico"))

        plt.subplot(111)
        plt.xlabel("x")
        plt.ylabel("y")

        for plot in plots:
            plt.plot(plot[0], plot[1], label=plot[2])
        plt.legend()

        plt.show()


class ODESolverApp:
    def __init__(self, diff_eq, solver, method):
        root = MainWindow().item
        graph_area = GraphArea(root).item

        pop_up = PopUpWindow(diff_eq)

        tabs = [
            TabSolution(graph_area, pop_up),
            TabLocalError(graph_area, pop_up),
            TabTotalError(graph_area, pop_up)
        ]

        for tab in tabs:
            graph_area.add(tab.item, text=tab.title)

        ControlPanel(root, tabs, diff_eq, solver, method)
        root.mainloop()


def ode_func(x, y):
    return 3*x*math.e**x - y*(1 - 1/x)


def ode_exact_sol(x, x0, y0):
    c = y0*((math.e**x0)/x0) - (3/2)*(math.e**(2*x0))
    return (3/2)*x*math.e**x + c*(x/math.e**x)


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


ode = DiffEquation(1, 0, 5, 5, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
ode_solver = ODESolver()
methods_list = [Exact(), Euler(), ModEuler(), RungeKutta()]

ODESolverApp(ode, ode_solver, methods_list)
