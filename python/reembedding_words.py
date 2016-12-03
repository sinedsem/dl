import glob

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

Fs = []
Ft = []

V_model = []
W_model = {}

for line in open("model.txt", encoding="utf-8"):
    splitted_line = line.strip("\n").split(" ")
    word = splitted_line[0]
    vec = np.array([float(x) for x in splitted_line[1:]])

    V_model.append(word)
    W_model[word] = vec

print(len(V_model))

V_data = []

for filename in glob.glob("data/text/*/*.txt"):
    for line in open(filename, encoding="utf-8"):
        for word in line.split(" "):
            V_data.append(word)

print(len(V_data))
V = np.intersect1d(V_model, V_data)
print(len(V))

M = len(V)

for word in V:
    vec = W_model[word]
    Fs.append(vec)
    Ft.append(np.zeros(len(vec)))

documents = []

N = len(Fs[0])

fi = np.ones(N)


def func(params, *args):
    Fs = args[0]
    documents = args[1]
    N = args[2]

    fi = params[:N]
    Ft = np.reshape(params[N:], (M, N))

    sum = 0
    for d in documents:
        for w in d:
            sum += np.math.log2(1 / (1 + np.math.exp(np.dot(-np.transpose(w), fi))))

    return sum - 0.3 * np.linalg.norm(Ft - Fs) ** 2


initial_values = np.append(fi, np.reshape(Ft, (1, N * M)))

a = fmin_l_bfgs_b(func, x0=initial_values, args=(Fs, documents, N), approx_grad=True)
print(a)

# mybounds = [(None, 2), (None, None)]
# fmin_l_bfgs_b(func, x0=initial_values, args=(x_true, y_true), bounds=mybounds, approx_grad=True)
