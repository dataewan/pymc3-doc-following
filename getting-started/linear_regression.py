import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of the dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate the outcome variable.
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

# plot the data

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X1")
axes[1].set_xlabel("X2")

plt.savefig("linear_regression_data.png")
plt.close()


basic_model = pm.Model()

with basic_model:
    # Normal prior distributions for the regression coefficients
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=10, shape=2)
    # Half normal prior for the standard deviation of the observations
    sigma = pm.HalfNormal("sigma", sd=1)

    # This is a deterministic variable that describes the linear relationship
    # between the regression coefficients.
    mu = alpha + beta[0] * X1 + beta[1] * X2

    Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=Y)


# finding the MAP estimate. Not recommended, only for historic reasons.
# map_estimate = pm.find_MAP(model=basic_model)


# taking samples from the model.
with basic_model:
    trace = pm.sample(500)


pm.traceplot(trace)
plt.savefig("traceplot.png")
plt.close()
