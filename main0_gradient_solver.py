from math import exp, log
import time


x_i, tau, alpha = 3, 0.4, 0.05


def gradient_fo(x):
    x1, b1, h1 = 2, 1, 1
    x2, b2, h2 = 6, 2, 2
    # return tau - 1 / (1 + exp((3 - z) / alpha))
    # return - tau + 1 / (1 + exp((x_i - x) / alpha))
    return - b1 + (b1 + h1) / (1 + exp((x1 - x) / alpha)) - b2 + (b2 + h2) / (1 + exp((x2 - x) / alpha))


x_str = x_i - alpha * log((1 - tau) / tau)
print(x_str)
print(gradient_fo(x_str))
print(gradient_fo(3))
print(gradient_fo(0))
print(gradient_fo(7))

x = 0
iteration_limit = 500
eta = 0.2
tolerance = 1e-5

start = time.process_time()
for i in range(iteration_limit):
    if i % 10 == 0:
        print(f'The value of x in iteration {i} is {x}')
    gradient_x = gradient_fo(x)
    if abs(gradient_x) < tolerance:
        break
    else:
        x -= eta * gradient_x
elapsed_time = time.process_time() - start

if i < iteration_limit:
    print(f'The solution is: {x}')
else:
    print('Max iterations achieved.')
print(f'Elapsed time: {elapsed_time} s')
