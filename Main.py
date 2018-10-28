import math
import numpy
import matplotlib.pyplot as plt


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
    return 3/2*x*(math.e**x - math.e**(2-x))


diff_eq = DiffEquation(1, 0, 5, 0.1, ode_func, ode_exact_sol, "y' = 3xe^x - y(1 - 1/x)")
solver = ODESolver()

m0 = solver.solve(diff_eq, Exact())
m1 = solver.solve(diff_eq, Euler())
m2 = solver.solve(diff_eq, ModEuler())
m3 = solver.solve(diff_eq, RungeKutta())

plot(m0, m1, m2, m3, diff_eq)
