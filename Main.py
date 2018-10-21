import math
import matplotlib.pyplot as plt


def euler_method(x0, y0, xn, h, f):
    xs = [x0]
    ys = [y0]

    while xs[-1] < xn:
        y = ys[-1] + h*f(xs[-1], ys[-1])
        x = xs[-1] + h
        ys.append(y)
        xs.append(x)

    return xs, ys


def mod_euler_method(x0, y0, xn, h, f):
    xs = [x0]
    ys = [y0]

    while xs[-1] < xn:
        x = xs[-1] + h
        k1 = f(xs[-1], ys[-1])
        k2 = f(x, ys[-1] + h*k1)
        y = ys[-1] + h*((k1 + k2)/2)
        ys.append(y)
        xs.append(x)

    return xs, ys


def runge_kutta(x0, y0, xn, h, f):
    xs = [x0]
    ys = [y0]

    while xs[-1] < xn:
        k1 = h*f(xs[-1], ys[-1])
        k2 = h*f(xs[-1] + h/2, ys[-1] + k1/2)
        k3 = h*f(xs[-1] + h/2, ys[-1] + k2/2)
        k4 = h*f(xs[-1] + h, ys[-1] + k3)
        xs.append(xs[-1] + h)
        ys.append(ys[-1] + k1/6 + k2/3 + k3/3 + k4/6)

    return xs, ys


def plot(d1, d2, d3, d4, d5, d6, formula):
    plt.plot(d1[0], d1[1], label="Euler, h = 0.1")
    plt.plot(d2[0], d2[1], label="Euler, h = 0.05")
    plt.plot(d3[0], d3[1], label="Modified-Euler, h = 0.1")
    plt.plot(d4[0], d4[1], label="Modified-Euler, h = 0.05")
    plt.plot(d5[0], d5[1], label="Runge-Kutta, h = 0.1")
    plt.plot(d6[0], d6[1], label="Runge-Kutta, h = 0.05")

    plt.title(formula)
    plt.axis([min(d1[0]), max(d1[0]), min(d1[1]), max(d1[1])])
    plt.legend()
    plt.show()


def print_table(diffs1, diffs2, formula):
    print("\n\n  ", formula, "\n|--------------------------------|")
    print("|  x  |   T1   |   T3   |   T5   |")
    # for coords in diffs1:
    #     for i in range(coords[0][0], coords[0][-1]):
    #         print('%0.2f' % coords[0][i], " ")
    # print("|  x  |  T2  |  T4  |  T6  |")
    # print()
    # for coords in diffs2:
    #     print('%0.2f' % coords[0], " ")


def diff_1(x, y):
    return 2*x*x + 3*y*y - 2


def diff_2(x, y):
    return y + math.sqrt(x*x + y*y)


def diff_3(x, y):
    return x*x - 3*x*y + y*y - 3*y


def diff_4(x, y):
    return (1 + x)/(1 - y*y)


diffs = [{"formula": "y' = 2x^2 + 3y^2 - 2",      "x0": 2, "y0": 1, "xn": 3, "func": diff_1},
         {"formula": "y' = y + sqrt(x^2 + y^2)",  "x0": 0, "y0": 1, "xn": 1, "func": diff_2},
         {"formula": "y' + 3y = x^2 - 3xy + y^2", "x0": 0, "y0": 2, "xn": 1, "func": diff_3},
         {"formula": "y' = (1 + x)/(1 - y^2)",    "x0": 2, "y0": 3, "xn": 3, "func": diff_4}]

for diff in diffs:
    d1 = euler_method(diff["x0"],     diff["y0"], diff["xn"], 0.1,  diff["func"])
    d2 = euler_method(diff["x0"],     diff["y0"], diff["xn"], 0.05, diff["func"])
    d3 = mod_euler_method(diff["x0"], diff["y0"], diff["xn"], 0.1,  diff["func"])
    d4 = mod_euler_method(diff["x0"], diff["y0"], diff["xn"], 0.05, diff["func"])
    d5 = runge_kutta(diff["x0"],      diff["y0"], diff["xn"], 0.1,  diff["func"])
    d6 = runge_kutta(diff["x0"],      diff["y0"], diff["xn"], 0.05, diff["func"])
    plot(d1, d2, d3, d4, d5, d6, diff["formula"])
    print_table([d1, d3, d5], [d2, d4, d6], diff["formula"])
