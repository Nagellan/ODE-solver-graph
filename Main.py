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
        root.iconbitmap(default="favicon.ico")
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
        sep.grid(column=0, columnspan=2, row=4, sticky=(tk.N, tk.W, tk.E, tk.S))

        ttk.Label(control_panel, text="n0")
        entry_n0 = ttk.Entry(control_panel)
        entry_n0.insert(tk.END, 2)

        ttk.Label(control_panel, text="n")
        entry_nn = ttk.Entry(control_panel)
        entry_nn.insert(tk.END, 10)

        i = 0
        for child in control_panel.winfo_children():
            if i == 4:
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

            # plots = []
            # for mtd in self.methods:
            #     plots.append(self.solver.solve(self.diff_eq, mtd))
            #
            # plots1 = []
            # for plot in plots[1:]:
            #     plots1.append((plot[0], list(abs(np.array(plot[1]) - np.array(plots[0][1]))), plot[2]))
            #
            # plots2 = []
            # ns = np.linspace(n0, nn, nn - 1)
            # for mtd in self.methods[1:]:
            #     errors = []
            #     plot = []
            #     for _n in ns:
            #         self.diff_eq.n = _n
            #         plot = self.solver.solve(self.diff_eq, mtd)
            #         exact = self.solver.solve(self.diff_eq, self.methods[0])
            #         errors.append(max(list(abs(np.array(plot[1]) - np.array(exact[1])))))
            #     plots2.append((ns, errors, plot[2]))
            #
            # self.tabs[0].update(plots)
            # self.tabs[1].update(plots1)
            # self.tabs[2].update(plots2)

            plots = self.tabs[0].get_solutions(self.diff_eq, self.solver, self.methods)
            self.tabs[0].update(plots)
            self.tabs[1].update(self.tabs[1].get_local_errors(plots))
            self.tabs[2].update(self.tabs[2].get_global_errors(self.diff_eq, self.solver, self.methods, n0, nn))

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
        graph_area = ttk.Notebook(root, padding="15", width=750, height=550)
        graph_area.pack(fill='both', expand='yes')
        graph_area.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.item = graph_area


class Tab:
    title = "Empty tab"

    def __init__(self, graph_area):
        tab = tk.Frame(graph_area, width=750, height=400)
        fig = Figure(figsize=(5, 5), dpi=100)

        graph = fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.item = tab
        self.graph = graph
        self.canvas = canvas
        self.toolbar = toolbar

    def update(self, plots):
        self.graph.clear()

        for plot in plots:
            self.graph.plot(plot[0], plot[1], label=self.title + " of " + plot[2])

        self.graph.legend()
        self.canvas.draw()
        self.toolbar.update()


class TabSolution(Tab):
    title = "Solution"

    def get_solutions(self, diff_eq, solver, methods):
        plots = []
        for mtd in methods:
            plots.append(solver.solve(diff_eq, mtd))
        return plots


class TabLocalError(Tab):
    title = "Local error"

    def get_local_errors(self, plots):
        local_errors = []
        for plot in plots[1:]:
            local_errors.append((plot[0], list(abs(np.array(plot[1]) - np.array(plots[0][1]))), plot[2]))
        return local_errors


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
        return global_errors


class ODESolverApp:
    def __init__(self, diff_eq, solver, method):
        root = MainWindow().item
        graph_area = GraphArea(root).item

        tabs = [
            TabSolution(graph_area),
            TabLocalError(graph_area),
            TabTotalError(graph_area)
        ]

        for tab in tabs:
            graph_area.add(tab.item, text=tab.title)

        ControlPanel(root, tabs, diff_eq, solver, method)
        root.mainloop()


def ode_func(x, y):
    return 3*x*math.e**x - y*(1 - 1/x)


def ode_exact_sol(x, x0, y0):
    c = y0*(math.e**x0/x0) - 3/2*math.e**(2*x0)
    return 3/2*x*math.e**x + c*(x/math.e**x)


ode = DiffEquation(1, 0, 5, 5, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
ode_solver = ODESolver()
methods_list = [Exact(), Euler(), ModEuler(), RungeKutta()]

ODESolverApp(ode, ode_solver, methods_list)
