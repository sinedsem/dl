import numpy as np
from scipy.optimize import fmin_l_bfgs_b

x_true = np.arange(0, 10, 0.1)
m_true = 2.5
b_true = 1.0
y_true = m_true * x_true + b_true

print(x_true)
print(y_true)


def func(params, *args):
    x = args[0]
    y = args[1]
    m, m1, m2, m3, b = params
    y_model = (m + m1 + m2 + m3) * x + b
    error = y - y_model
    return sum(error ** 2)


initial_values = np.array([0.25, 0.25, 0.25, 0.25, 0.0])

a = fmin_l_bfgs_b(func, x0=initial_values, args=(x_true, y_true), approx_grad=True)
print(a)

# mybounds = [(None, 2), (None, None)]
# fmin_l_bfgs_b(func, x0=initial_values, args=(x_true, y_true), bounds=mybounds, approx_grad=True)
