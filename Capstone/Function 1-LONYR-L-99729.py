import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.stats import norm

# STEP 1: Load the model from the File Explorer.
X = np.load(r'C:\Users\A.okenyi\OneDrive - Shell\Documents\Professional Certificate in ML & AI\initial_data\function_1\initial_inputs.npy')
y = - np.load(r'C:\Users\A.okenyi\OneDrive - Shell\Documents\Professional Certificate in ML & AI\initial_data\function_1\initial_outputs.npy')

X = np.append(X, np.array([
    [0.595959, 0.626262],
    [0.5, 0.5],
    [0.492588, 0.485863],
    [0.576014, 0.602462],
    [0.601155, 0.610645],
    [0.591697, 0.617293]
                         ]), axis=0)

y = np.append(y, np.array([
        0.0955520425802805,
        2.67528799107424E-09,
        3.23249270578001E-07,
        -0.00207640294539396,
        0.124649827387687,
        0.0162506399153848
                         ]), axis=0)

dims = X.shape[1] # The number of dimensions of X
af_option = 'UB'
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

num_pred = 50
x1 = np.array([np.random.uniform() for i in range(num_pred)])
x2 = np.array([np.random.uniform() for i in range(num_pred)])

x_pred = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
y_mean, y_std = model.predict(x_pred, return_std=True)

if dims <= 2:
    
    # Plot the mean results
    x1_plt, x2_plt = np.meshgrid(x1, x2)
    y_plt = y_mean.reshape(num_pred, num_pred)
    plt.pcolormesh(x1_plt, x2_plt, y_plt, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.scatter(x2_scatter, x1_scatter, facecolor='w', edgecolors='k')
    plt.title('Mean Gaussian')
    plt.show()
    plt.clf()

    # Plot the std results
    y_plt = y_std.reshape(num_pred, num_pred)
    plt.pcolormesh(x1_plt, x2_plt, y_plt, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.scatter(x2_scatter, x1_scatter, facecolor='w', edgecolors='k')
    plt.title('Std Gaussian')
    plt.show()
    plt.clf()

# STEP 4: Acquisition function and obtain next point
explor = 0
numerator = max(y) * explor

if af_option == 'PI': # Probability of Improvement (PI)
    af = norm.cdf(numerator, y_mean, y_std)

elif af_option == 'UB': # Upper Bound
    af = y_mean + 3 * y_std

elif af_option == 'EI': # Expected Improvement (EI)
    af = (y_mean - numerator) * norm.cdf(numerator, y_mean, y_std) + y_std * norm.pdf(numerator, y_mean, y_std)

next_point = x_pred[np.argmax(af)]
print(next_point)

if dims <= 2:

    x1_plt, x2_plt = np.meshgrid(x1, x2)
    y_plt = af.reshape(num_pred, num_pred)

    # Plot the acquistion function
    plt.pcolormesh(x1_plt, x2_plt, y_plt, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.scatter(x2_scatter, x1_scatter, facecolor='w', edgecolors='k')
    plt.scatter(next_point[1], next_point[0], facecolor='k', edgecolors='k')
    plt.title('Acquistion Function: ' + af_option)
    plt.show()
    plt.clf