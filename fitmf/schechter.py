import numpy as np
from scipy.special import gamma, gammainc, gammaincinv


def function(M, theta):
    alpha, Mstar = theta
    return (1 / Mstar) * np.power(M / Mstar, alpha) * np.exp(-M / Mstar)


def CDF(M, theta):
    alpha, Mstar = theta
    return gammainc(alpha + 1, M / Mstar)


def invCDF(X, theta):
    alpha, Mstar = theta
    return Mstar * gammaincinv(alpha + 1, X)


def normalized_integral(theta, Mlim):
    alpha, Mstar = theta
    return (1 - CDF(Mlim, theta)) * gamma(alpha + 1)


def simulate(N, theta, Mlim):
    randomU = CDF(Mlim, theta) + (1 - CDF(Mlim, theta)) * np.random.rand(N)
    return invCDF(randomU, theta)
