import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

n = 40
h = 1./n
dx = 1./n
dt = .1/n
mu = dt/(dx*dx)

def uex(x,y):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def solve():
    #Build tridiagonal
    D = np.diag(np.ones(n*n)*-2*mu) + np.diag(np.ones(n*n-1)*mu,1) + np.diag(np.ones(n*n-1)*mu,-1)
    I = np.identity(n*n)
    u = np.zeros((n*n))
    #build our vector u representing the 2-D field
    for i in range(1,n-1):
        for j in range(1,n-1):
            u[index(i,j)] = uex(i,j)
    for l in range(n):
        #step 1
        b = np.dot((I + (dt/2)*D),u)
        A = (I - (dt/2)*D)
        u_star = np.linalg.solve(A,b)

        #step 2
        b_2 = np.dot((I + (dt/2)*np.square(D)),u_star)
        u = np.linalg.solve(A,b_2)
    uu = np.reshape(u,(n,n))
    print(uu)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X,Y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    ax.plot_surface(X,Y,uu, cmap=cm.Blues)
    plt.show()

def index(i,j):
    return i*n + j
solve()
