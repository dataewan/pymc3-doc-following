Working through the pymc documentation.

https://docs.pymc.io/


# Notes

## [Getting started](https://docs.pymc.io/notebooks/getting_started.html)

Example about [linear regression](getting_started/linear_regression.py).


### Specifying the model

Introduced the **prior distributions** for the regression coefficients.
Using `Normal` distributions.
There is a distribution for the standard deviation of the observations,
which is a `HalfNormal`, so that it is always positive.

Showing a **deterministic** variable, that is a combination of the parent variables.
Using maths to describe a linear relationship.

There is the **observed** distribution,
using a Normal distribution again.
Where the mean is the deterministic variable,
and the SD of the observations is the half normal from above.


### Fitting the model

Two approaches,
which of them is appropriate depends on the structure of the model and the needs of the analysis.

You can find the _maximum a posteriori_ point using optimisation methods (MAP).
The MAP is the mode of the posterior distribution.
It is easy to do, but won't represent the whole probability distribution.
They provide the `find_MAP()` function mainly for historic reasons.
It isn't that useful, and can usually only find a local optimum.

Preferred option is to compute summaries based on samples drawn from the posterior distribution using Markov Chain Monte Carlo (MCMC) methods.

Sampling will try and select samples from a distribution that approximates the true posterior distribution.
Different step methods for the MCMC will work for different kinds of distributions.
You can either assign them manually, or you can rely on auto assignment.
If you're using auto assignment, then you get this result:

 - Binary variables will be assigned to `BinaryMetropolis`
 - Discrete variables will be assigned to `Metropolis`
 - Continuous variables will be assigned to `NUTS`

pymc will also initialise the setp function to sensible values if left to itself.


```python
# take 500 samples from the model
with basic_model:
  trace = pm.sample(500)
```


### Posterior analysis

The traceplot is the way that you start analysis of the posterior samples.

```python
pm.traceplot(trace)
```

![example traceplot](getting_started/traceplot.png)

Also the output from `pm.summary` is useful.

```python
pm.summary(trace).round(2)
```

           mean    sd    mc_error    hpd_2.5    hpd_97.5    n_eff    Rhat
-------  ------  ----  ----------  ---------  ----------  -------  ------
alpha      0.91  0.1         0          0.71        1.1   1160.38       1
beta__0    0.95  0.09        0          0.75        1.12  1815.64       1
beta__1    2.63  0.52        0.01       1.67        3.67  1409.75       1
sigma      0.99  0.07        0          0.87        1.15  1636.45       1



