### The steepest descent method with Armijo's Rule with an example ###

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


# Define a quadratic, multivariable objective function
'''
The steepest decent method summary:
1. Choose a vector of decision variable x and a stopping accuracy ε > 0 as well as δ between (0, 1/2)
2. If the norm of gradient decent is less than stopping accuracy ε, then stop. Otherwise, proceed.
3. Let d = - gradient decent, d is for stepping direction.
4. Let t = 1.
5. Check if f(x+td) < f(x)+transpose(tδ(∇f(x)))*d holds. i.e f(x+td) < f(x) + Square Gradient length
6. If not, redefine t to t:=1/2 and go to step 5 again.
5. Redefine x:=x+td.
6. Goto step 2.
'''


def f(x):
    '''Object function'''
    return x[0] ** 2 - 4 * x[0] + x[1] ** 2 - x[1] - x[0] * x[1]


def g(x):
    '''Gradient of the objective function, here we have gradients or partial
    derivative of f with respect to xi, in this case, xi=(x1,x2)
    '''
    return np.array([2 * x[0] - 4 - x[1], 2 * x[1] - 1 - x[0]])


def square_norm_gradient(x):
    '''Calculatint the norm of gradient of f(x)'''
    return x[0] ** 2 + x[1] ** 2

x = np.array([-1, 4]) # initial value of x
stop_accuracy = 0.01
delta = 0.4
t = 1  # self-defined step = 1
iteration = 1

gradient_norm = [sqrt(square_norm_gradient(g(x)))]
iterations = [iteration]
while sqrt(square_norm_gradient(g(x))) >= stop_accuracy:
    d = -g(x) # g(x) is the gradient of f(x)
    df = f(x + t * d)
    if df <= f(x) + t * delta * square_norm_gradient(x):  # check the step of length t
        print(f'Step: {iteration} | x: {x} | Gradient: {g(x)} | Gradient length: {sqrt(square_norm_gradient(g(x)))} | t: {t} ')
        gradient = g(x + t * d)
        d = -g(x)
        x = x + t * d # redefine x
        iteration += 1 # track steps
        iterations.append(iteration)
        gradient_norm.append(sqrt(square_norm_gradient(g(x))))
    else:  # if the step length is too large, set t= t/2 and then check if f(x+td) < f(x) + Square Gradient length holds
        t = t / 2
        continue

# plot gradient norm VS iterations
plt.plot(iterations, gradient_norm)
plt.ylabel("Length of Gradient Norm")
plt.show()
