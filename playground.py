import numpy as np
import random
import matplotlib.pyplot as plt

xs = np.arange(-1, 1, 0.1)
ys = []
def f(x):
    return 5*(x**2)


real_ys = [f(x) for x in xs]
ys_1 = [f(x)+0.2 for x in xs]
ys_2 = [f(x)-0.2 for x in xs]

rand_arr = random.sample(range(0, len(xs)), int(len(xs)/2))
for i in range(len(xs)):
    if i in rand_arr:
        ys += [ys_1[i]]
    else:
        ys += [ys_2[i]]

plt.figure()
plt.scatter(xs, ys)
plt.plot(xs, ys)
plt.show()

def f_grad(x, w, y):
    return (x.T @ w - y) @ x

w_0 = random.sample(range(0, 1, 0.001), 3)      # ax^2 + bx + c

data = xs[:]            # copy of xs.
