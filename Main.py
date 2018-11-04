import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg


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
        root = Tk()

        root.title("ODE Solver")
        root.resizable(width=False, height=False)
        root.iconbitmap(default="favicon.ico")

        return root


class ControlPanel:
    def __init__(self, root, diff_eq, solver, method):
        self.root = root
        self.diff_eq = diff_eq
        self.solver = solver
        self.method = method

    def create(self):
        control_panel = ttk.Frame(self.root, padding="15")
        control_panel.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(control_panel, text="x0")
        entry_x0 = ttk.Entry(control_panel)
        entry_x0.insert(END, self.diff_eq.x0)

        ttk.Label(control_panel, text="y0")
        entry_y0 = ttk.Entry(control_panel)
        entry_y0.insert(END, self.diff_eq.y0)

        ttk.Label(control_panel, text="xn")
        entry_xn = ttk.Entry(control_panel)
        entry_xn.insert(END, self.diff_eq.xn)

        ttk.Label(control_panel, text="n")
        entry_n = ttk.Entry(control_panel)
        entry_n.insert(END, self.diff_eq.n)

        btn = ttk.Button(control_panel, text="Plot",
                         command=lambda e_x0=entry_x0, e_y0=entry_y0, e_xn=entry_xn, e_n=entry_n:
                         self.update(int(e_x0.get()), int(e_y0.get()), int(e_xn.get()), int(e_n.get())))
        btn.grid(column=0, columnspan=2, row=5, sticky=(W, E))

        i = 0
        for child in control_panel.winfo_children():
            child.grid_configure(padx=5, pady=5)

            if isinstance(child, ttk.Label):
                child.grid(column=0, row=i, sticky=(W, E))

            if isinstance(child, ttk.Entry):
                child.config(justify=CENTER)
                child.grid(column=1, row=i, sticky=(W, E))
                i = i + 1

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

            self.plot(plots)

    def plot(self, plots):
        plt.figure(figsize=(6, 7))
        plt.suptitle(self.diff_eq.formula + "\nx0 = " + str(self.diff_eq.x0) + ", xn = " + str(self.diff_eq.xn)
                     + ", y0 = " + str(self.diff_eq.y0) + ", n = " + str(self.diff_eq.n))

        plt.subplot(211)
        plt.title("Solutions")
        plt.xlabel("x")
        plt.ylabel("y")

        if len(plots) > 0:
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


class ODESolverApp:
    def __init__(self, diff_eq, solver, method):
        root = MainWindow().create()
        ControlPanel(root, diff_eq, solver, method).create()
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
