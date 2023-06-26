import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.stats import norm

# STEP 1: Load the model from the File Explorer.
X = np.load(r'C:\Users\A.okenyi\OneDrive - Shell\Documents\Professional Certificate in ML & AI\initial_data\function_4\initial_inputs.npy')
y = np.load(r'C:\Users\A.okenyi\OneDrive - Shell\Documents\Professional Certificate in ML & AI\initial_data\function_4\initial_outputs.npy')

X = np.append(X, np.array([
    [0.285714, 0.857142, 0.979591, 0.938775],
    [0.286443, 0.823585, 0.846135, 0.97612],
    [0.273927, 0.850303, 0.98227,  0.962445],
    [0.25029, 0.802122, 0.974834, 0.976119],
    [0.253462, 0.824737, 0.958529, 0.957297],
    [0.269793, 0.795867, 0.99461, 0.990151]
                         ]), axis=0)

y = np.append(y, np.array([
        2204.55729126245,
        1415.358315,
        2387.252095037,
        2164.34008795898,
        1980.50910493224,
        2458.37138701986
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

    x_pred = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, dims)
    y_mean, y_std = model.predict(x_pred, return_std=True)

    # STEP 4: Acquisition function and obtain next point
    explor = 0
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


