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


class Method:
    def euler(self, diff_eq):
        xs = [diff_eq.x0]
        ys = [diff_eq.y0]

        while xs[-1] < diff_eq.xn:
            y = ys[-1] + diff_eq.h*diff_eq.f(xs[-1], ys[-1])
            x = xs[-1] + diff_eq.h

            ys.append(y)
            xs.append(x)

        return xs, ys, "Euler method"

    def mod_euler(self, diff_eq):
        xs = [diff_eq.x0]
        ys = [diff_eq.y0]

        while xs[-1] < diff_eq.xn:
            x = xs[-1] + diff_eq.h
            k1 = diff_eq.f(xs[-1], ys[-1])
            k2 = diff_eq.f(x, ys[-1] + diff_eq.h*k1)
            y = ys[-1] + diff_eq.h*((k1 + k2)/2)

            ys.append(y)
            xs.append(x)

        return xs, ys, "Modified Euler method"

    def runge_kutta(self, diff_eq):
        xs = [diff_eq.x0]
        ys = [diff_eq.y0]

        while xs[-1] < diff_eq.xn:
            k1 = diff_eq.h*diff_eq.f(xs[-1], ys[-1])
            k2 = diff_eq.h*diff_eq.f(xs[-1] + diff_eq.h/2, ys[-1] + k1/2)
            k3 = diff_eq.h*diff_eq.f(xs[-1] + diff_eq.h/2, ys[-1] + k2/2)
            k4 = diff_eq.h*diff_eq.f(xs[-1] + diff_eq.h, ys[-1] + k3)

            xs.append(xs[-1] + diff_eq.h)
            ys.append(ys[-1] + k1/6 + k2/3 + k3/3 + k4/6)

        return xs, ys, "Runge Kutta method"


class ODEsolver:
    def solve(self, diff_eq, method):
        return method(diff_eq)


def plot(m1, m2, m3, diff_eq):
    plt.plot(m1[0], m1[1], label=m1[2])
    plt.plot(m2[0], m2[1], label=m2[2])
    plt.plot(m3[0], m3[1], label=m3[2])

    plt.title(diff_eq.formula + ", h = " + str(diff_eq.h))
    plt.axis([min(m1[0]), max(m1[0]), min(m1[1]), max(m1[1])])
    plt.legend()
    plt.show()


def function_15(x, y):
    return 3*x*math.e - y*(1 - 1/x)


ode_15 = DiffEquation(1, 0, 5, 1, function_15, "y' = 3xe^x - y(1 - 1/x)")
solver = ODEsolver()
method = Method()

m1 = solver.solve(ode_15, method.euler)
m2 = solver.solve(ode_15, method.mod_euler)
m3 = solver.solve(ode_15, method.runge_kutta)

plot(m1, m2, m3, ode_15)
