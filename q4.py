import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

h = 22.5
P = np.ones((4,4))
b = 5500
k = (h*h)/(6*b)
l = 73
w = 160
t = 100
C = (58,147)
Q = (58,103)
T = (31,14)
fac = [C,Q,T]
def buildData(f1):
    d = []
    for line in f1:
        list1 = [int(number) for number in line.split(' ')]
        d.append(list1)
    return d

def solve():
    f = open('van_vleck.txt', 'r')
    vv = np.array(buildData(f))
    m,n = vv.shape
    u = np.zeros(m*n)
    enforce_pizza_zone(u)
    #build discretization matrix
    D = np.diag(np.ones(m*n)*-4)
    for i in range(m):
        for j in range(n):
            ij = ind(i,j)
            if in_range(i+1,j):
                if vv[i+1,j] == 1:
                    D[ij,ij] += 1
                else:
                    D[ij,ind(i+1,j)] = 1
            if in_range(i,j+1):
                if vv[i, j+1] == 1:
                    D[ij, ij] += 1
                else:
                    D[ij,ind(i,j+1)] = 1
            if in_range(i-1,j):
                if vv[i - 1, j] == 1:
                    D[ij, ij] += 1
                else:
                    D[ij,ind(i-1,j)] = 1
            if in_range(i,j-1):
                if vv[i, j-1] == 1:
                    D[ij, ij] += 1
                else:
                    D[ij,ind(i,j-1)] = 1
    D*=(b/(h*h))
    times = (1,5,25,100)
    curr = 0
    iter = 0
    while iter < t:
        u_new = np.dot(D,u)
        u_new*=k
        u = u_new + u
        enforce_pizza_zone(u)
        iter+=k
        for teach in fac:
            if u[ind(teach[0], teach[1])] > 1e-4:
                print("Professor at " + str(teach) + " smells pizza! The time is " + str(iter))
                fac.remove(teach)
        if iter >= times[curr]:
            uu = np.reshape(np.power(u,.25) , (l,w))
            for i in range(l):
                for j in range(w):
                    if(vv[i,j] == 1):
                        uu[i,j] = 0
            plt.imshow(uu, cmap='hot', interpolation='nearest')
            plt.title("time = " + str(times[curr]))
            plt.show()
            curr+=1

def in_range(i,j):
    return i < 73 and j < 160 and i*w + j >= 0

def ind(i,j):
    return i*w+j

def enforce_pizza_zone(u):
    for i in range(36,40):
        for j in range(44,48):
            u[ind(i,j)] = 1

solve()