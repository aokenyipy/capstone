import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.stats import norm

# STEP 1: Load the model from the File Explorer.
X = np.load(r'C:\Users\A.okenyi\OneDrive - Shell\Documents\Professional Certificate in ML & AI\initial_data\function_7\initial_inputs.npy')
y = np.load(r'C:\Users\A.okenyi\OneDrive - Shell\Documents\Professional Certificate in ML & AI\initial_data\function_7\initial_outputs.npy')

X = np.append(X, np.array([
    [0.0000, 0.4444, 0.2222, 0.2222, 0.3333, 0.7778],
    [0.054562, 0.377748, 0.247517, 0.270995, 0.327662, 0.90223],
    [0.036548, 0.422654, 0.259953, 0.135579, 0.287203, 0.795534],
    [0.010908, 0.360313, 0.235805, 0.215896, 0.382804, 0.806458],
    [0.012305, 0.37838, 0.185785, 0.281644, 0.309703, 0.702902],
    [0.009392, 0.317647, 0.144952, 0.255016, 0.318602, 0.636865]
                         ]), axis=0)

y = np.append(y, np.array([
        1.58275434964637,
        1.43261691318864,
        1.45063457851754,
        1.62109706820207,
        2.04679224555022,
        2.13657654903849
                         ]), axis=0)

dims = X.shape[1] # The number of dimensions of X
af_option = 'EI'

if dims <= 2:
    x1_scatter = [i[0] for i in X]
    x2_scatter = [i[1] for i in X]

# STEP 2: Obtain the optimal kernel

# Step 2a: Obtain the log-marginal-likelihood for optimal RBF kernel
rbf_kernel = (1 ** 2) * RBF(length_scale=1)
model = GaussianProcessRegressor(kernel=rbf_kernel, n_restarts_optimizer=25)
model.fit(X, y)
opt_rbf = model.kernel_
model = GaussianProcessRegressor(kernel=opt_rbf)
model.fit(X, y)
rbf_lml = model.log_marginal_likelihood()

# Step 2b: Obtain the log-marginal-likelihood for optimal Matern kernel
mat_options = [3/2, 5/2]
mat_lml = []
mat_ker = []

for m in mat_options:
    mat_kernel = (1 ** 2) * Matern(length_scale=1, nu=m)
    model = GaussianProcessRegressor(kernel=mat_kernel, n_restarts_optimizer=25)
    model.fit(X, y)
    mat_ker.append(model.kernel_)
    model = GaussianProcessRegressor(kernel=model.kernel_)
    model.fit(X, y)
    mat_lml.append(model.log_marginal_likelihood())

# Step 2c: Identify opitmal kernel
if rbf_lml > max(mat_lml):
    optimal = opt_rbf
    print('RBF is optimal')
    print(optimal)
else:
    optimal = mat_ker[mat_lml.index(max(mat_lml))]
    print('Matern is optimal')
    print(optimal)

# STEP 3: Run model again
model = GaussianProcessRegressor(kernel=optimal)
model.fit(X, y)

points = []
values = []

for j in range(200):

    num_pred = 5
    x1 = np.array([np.random.uniform() for i in range(num_pred)])
    x2 = np.array([np.random.uniform() for i in range(num_pred)])
    x3 = np.array([np.random.uniform() for i in range(num_pred)])
    x4 = np.array([np.random.uniform() for i in range(num_pred)])
    x5 = np.array([np.random.uniform() for i in range(num_pred)])
    x6 = np.array([np.random.uniform() for i in range(num_pred)])

    x_pred = np.array(np.meshgrid(x1, x2, x3, x4, x5, x6)).T.reshape(-1, dims)
    y_mean, y_std = model.predict(x_pred, return_std=True)

    # STEP 4: Acquisition function and obtain next point
    explor = 0.05
    numerator = max(y) + explor

    if af_option == 'PI': # Probability of Improvement (PI)
        af = norm.cdf(numerator, y_mean, y_std)

    elif af_option == 'UB': # Upper Bound
        af = y_mean + explor * y_std

    elif af_option == 'EI': # Expected Improvement (EI)
        af = (y_mean - numerator) * norm.cdf(numerator, y_mean, y_std) + y_std * norm.pdf(numerator, y_mean, y_std)

    next_point = x_pred[np.argmax(af)]

    points.append(next_point)
    values.append(max(af))

next_point = points[np.argmax(values)]

print(next_point)