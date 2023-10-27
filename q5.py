import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def uex(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def solve(n, times):
    h = 1. / n
    dx = 1. / n
    dt = .1 / n
    mu = 1 / (dx * dx)
    # Build tridiagonal
    D = np.diag(np.ones(n) * -2 * mu) + np.diag(np.ones(n - 1) * mu, 1) + np.diag(np.ones(n - 1) * mu, -1)
    I = np.identity(n)
    u = np.zeros((n, n))
    u_star = np.zeros((n, n))
    # build our vector u representing the 2-D field
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            u[i, j] = uex(i, j)
    for l in range(n):
        A = (I - (dt / 2) * D)
        # step 1
        for i in range(n):
            b = u[:, i] + (dt / 2) * np.dot(D, u[:, i])
            u_star[:, i] = np.linalg.solve(A, b)

        # step 2
        for j in range(n):
            b_2 = u_star[j, :] + (dt / 2) * np.dot(D, u_star[j, :])
            u[j, :] = np.linalg.solve(A, b_2)
        if l in times:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
            ax.plot_surface(X, Y, u, cmap=cm.Blues)
            plt.title("time = " + str(l + 1) + ", h = " + str(h))
            plt.show()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    ax.plot_surface(X, Y, u, cmap=cm.Blues)
    plt.title("time = " + str(l + 1) + ", h = " + str(h))
    plt.show()


solve(100, (9, 24, 49, 99))

#n_sizes = (40, 80, 160)
#for n in n_sizes:
    #solve(n,[39])
