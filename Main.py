import math
import numpy
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
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

        ttk.Label(control_panel, text="h")
        entry_h = ttk.Entry(control_panel)
        entry_h.insert(END, self.diff_eq.h)

        btn = ttk.Button(control_panel, text="Plot",
                         command=lambda e_x0=entry_x0, e_y0=entry_y0, e_xn=entry_xn, e_h=entry_h:
                         self.update(e_x0, e_y0, e_xn, e_h))
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

    def update(self, x0, y0, xn, h):
        try:
            self.diff_eq.x0 = int(x0.get())
            self.diff_eq.y0 = int(y0.get())
            self.diff_eq.xn = int(xn.get())
            self.diff_eq.h = float(h.get())

            s0 = self.solver.solve(self.diff_eq, method["Exact"])
            s1 = self.solver.solve(self.diff_eq, method["Euler"])
            s2 = self.solver.solve(self.diff_eq, method["ModEuler"])
            s3 = self.solver.solve(self.diff_eq, method["RungeKutta"])

            self.plot(s0, s1, s2, s3)
        except ValueError:
            pass

    def plot(self, m0, m1, m2, m3):
        plt.figure(figsize=(6, 7))
        plt.suptitle(self.diff_eq.formula + ", h = " + str(self.diff_eq.h))

        plt.subplot(211)
        plt.title("Solutions")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.plot(m0[0], m0[1], label=m0[2])
        plt.plot(m1[0], m1[1], label=m1[2])
        plt.plot(m2[0], m2[1], label=m2[2])
        plt.plot(m3[0], m3[1], label=m3[2])
        plt.subplots_adjust(hspace=0.5)
        plt.legend()

        plt.subplot(212)
        plt.title("Errors")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.plot(m1[0], numpy.array(m1[1]) - numpy.array(m0[1]), label="Error of " + m1[2])
        plt.plot(m2[0], numpy.array(m2[1]) - numpy.array(m0[1]), label="Error of " + m2[2])
        plt.plot(m3[0], numpy.array(m3[1]) - numpy.array(m0[1]), label="Error of " + m3[2])
        plt.legend()

        plt.show()


class ODESolverApp:
    def __init__(self, diff_eq, solver, method):
        root = MainWindow().create()
        ControlPanel(root, diff_eq, solver, method).create()

        root.mainloop()


def ode_func(x, y):
    return 3*x*math.e**x - y*(1 - 1/x)


def ode_exact_sol(x):
    return 3/2*x*(math.e**x - math.e**(2 - x))


ode = DiffEquation(1, 0, 5, 1, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
solver = ODESolver()
method = {"Exact": Exact(), "Euler": Euler(), "ModEuler": ModEuler(), "RungeKutta": RungeKutta()}

ODESolverApp(ode, solver, method)
