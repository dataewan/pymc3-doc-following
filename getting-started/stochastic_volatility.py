from pandas_datareader import data
from matplotlib import pyplot as plt
import pymc3 as pm


returns = data.get_data_yahoo("SPY", start="2008-05-01", end="2009-12-01")[
    "Close"
].pct_change()


returns.plot(figsize=(10, 6))
plt.ylabel("daily returns in %")
plt.savefig("daily_returns.png")
plt.close()


with pm.Model() as sp500_model:
    nu = pm.Exponential("nu", 1 / 10, testval=5)
    sigma = pm.Exponential("sigma", 1 / 0.02, testval=.1)

    s = pm.GaussianRandomWalk("s", sd=sigma, shape=len(returns))

    volatility_process = pm.Deterministic(
        "volatility_process", pm.math.exp(-2 * s) ** 0.5
    )

with sp500_model:
    trace = pm.sample(2000)

pm.traceplot(trace, varnames=["nu", "sigma"])
plt.savefig("stochastic_volatility_traceplot.png")
