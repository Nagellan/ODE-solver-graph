import math
import matplotlib.pyplot as plt


class DiffEquation:
    def __init__(self, x0, y0, xn, h, f, formula):
        self.x0 = x0
        self.y0 = y0
        self.xn = xn
        self.h = h
        self.f = f
        self.formula = formula


class Solver:
    x0 = 0
    y0 = 0
    xn = 0
    h = 0
    f = None

    def solve(self, diff_eq, method):
        self.x0 = diff_eq.x0
        self.y0 = diff_eq.y0
        self.xn = diff_eq.xn
        self.h = diff_eq.h
        self.f = diff_eq.f

        return method()

    def euler_method(self):
        xs = [self.x0]
        ys = [self.y0]

        while xs[-1] < self.xn:
            y = ys[-1] + self.h*self.f(xs[-1], ys[-1])
            x = xs[-1] + self.h
            ys.append(y)
            xs.append(x)

        return xs, ys, "Euler method, h = "

    def mod_euler_method(self):
        xs = [self.x0]
        ys = [self.y0]

        while xs[-1] < self.xn:
            x = xs[-1] + self.h
            k1 = self.f(xs[-1], ys[-1])
            k2 = self.f(x, ys[-1] + self.h*k1)
            y = ys[-1] + self.h*((k1 + k2)/2)
            ys.append(y)
            xs.append(x)

        return xs, ys, "Modified Euler method, h = "

    def runge_kutta(self):
        xs = [self.x0]
        ys = [self.y0]

        while xs[-1] < self.xn:
            k1 = self.h*self.f(xs[-1], ys[-1])
            k2 = self.h*self.f(xs[-1] + self.h/2, ys[-1] + k1/2)
            k3 = self.h*self.f(xs[-1] + self.h/2, ys[-1] + k2/2)
            k4 = self.h*self.f(xs[-1] + self.h, ys[-1] + k3)
            xs.append(xs[-1] + self.h)
            ys.append(ys[-1] + k1/6 + k2/3 + k3/3 + k4/6)

        return xs, ys, "Runge Kutta method"


def plot(m1, m2, m3, diff_eq):
    plt.plot(m1[0], m1[1], label=m1[2])
    plt.plot(m2[0], m2[1], label=m2[2])
    plt.plot(m3[0], m3[1], label=m3[2])

    plt.title(diff_eq.formula + ", h = " + str(diff_eq.h))
    plt.axis([min(m1[0]), max(m1[0]), min(m1[1]), max(m1[1])])
    plt.legend()
    plt.show()


def var_15_func(x, y):
    return 3*x*math.e - y*(1 - 1/x)


var_15 = DiffEquation(1, 0, 5, 1, var_15_func, "y' = 3xe^x - y(1 - 1/x)")
solver = Solver()

m1 = solver.solve(var_15, solver.euler_method)
m2 = solver.solve(var_15, solver.mod_euler_method)
m3 = solver.solve(var_15, solver.runge_kutta)

plot(m1, m2, m3, var_15)
