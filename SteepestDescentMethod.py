### The steepest descent method with an example ###

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


# Define a quadratic, multivariable objective function
def f(x):
    '''Object function'''
    return x[0] ** 2 - 4 * x[0] + x[1] ** 2 - x[1] - x[0] * x[1]


def g(x):
    '''Gradient of the objective function'''
    return np.array([2 * x[0] - 4 - x[1], 2 * x[1] - 1 - x[0]])


def norm_gradient(x):
    '''Calculatint the norm of gradient of f(x)'''
    return sqrt(x[0] ** 2 + x[1] ** 2)


x = np.array([-1, 4]) # initial value of x
stop_accuracy = 0.01
t = 1 # self-defined step = 1
iteration = 1

gradient_norm = [norm_gradient(g(x))]
iterations = [iteration]

while norm_gradient(g(x)) >= stop_accuracy:
    d = -g(x)
    df = f(x + t * d)
    if df <= f(x):
        print(f'Step: {iteration} | x: {x} | Gradient: {g(x)} | Gradient length: {norm_gradient(g(x))} | t: {t} ')
        gradient = g(x + t * d)
        d = -g(x)
        x = x + t * d
        iteration += 1
        iterations.append(iteration)
        gradient_norm.append(norm_gradient(g(x)))
    else:
        t = t / 2
        continue

plt.plot(iterations, gradient_norm)
plt.show()
