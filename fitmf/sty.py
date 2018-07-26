import numpy as np
import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import gamma, gammainc, gammaincinv, gammaincc
import emcee
import corner

pdffile = PdfPages('gamma.pdf')


#-----------simplest example


def f(x):
    return np.exp(-x)


def CDF(x):
    return 1 - np.exp(-x)


def invCDF(X):
    return -np.log(1 - X)


xlim = 2.5
N = 10000
norm = 1 - CDF(xlim)

randomU = CDF(xlim) + (1 - CDF(xlim)) * np.random.rand(N)
xi = invCDF(randomU)
hist, bins = np.histogram(xi, bins=np.arange(0, 10.1, .5))
hist = hist / N
hist = hist * norm / np.diff(bins)
mids = .5 * (bins[1:] + bins[:-1])

fig = pp.figure(1)
sp1 = fig.add_subplot(1, 1, 1)
xdata = np.logspace(-2, 1, 200)
pp.plot(xdata, f(xdata), '-k', lw=1.5)
pp.axvline(x=xlim, ls='dotted', color='black')
pp.xlabel(r'$x$')
pp.ylabel(r'$f(x)$')
pp.errorbar(mids, hist, marker='.', ls='None', xerr=.25, ecolor='black', mec='black', mfc='black')

pp.savefig(pdffile, format='pdf')


#-----------log axes on plot


def f(x):
    return np.exp(-x)


def CDF(x):
    return 1 - np.exp(-x)


def invCDF(X):
    return -np.log(1 - X)


xlim = np.power(10, .3)
N = 10000
norm = 1 - CDF(xlim)

randomU = CDF(xlim) + (1 - CDF(xlim)) * np.random.rand(N)
xi = invCDF(randomU)
bins = np.power(10, np.arange(0, 2.01, .1))
hist, bins = np.histogram(xi, bins=bins)
hist = hist / N
hist = hist * norm / np.diff(bins)
mids = np.power(10, .5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))

fig = pp.figure(2)
sp1 = fig.add_subplot(1, 1, 1)
xdata = np.logspace(0, 1, 200)
pp.plot(np.log10(xdata), np.log10(f(xdata)), '-k', lw=1.5)
pp.axvline(x=np.log10(xlim), ls='dotted', color='black')
pp.xlabel(r'$\log_{10}x$')
pp.ylabel(r'$\log_{10}f(x)$')
pp.errorbar(np.log10(mids), np.log10(hist), marker='.', ls='None', xerr=.05, ecolor='black', mec='black', mfc='black')

pp.savefig(pdffile, format='pdf')


#-----------schechter function, dimensionless


def f(x, a):
    return np.power(x, a - 1) * np.exp(-x)


def CDF(x, a):
    return gammainc(a, x)


def invCDF(X, a):
    return gammaincinv(a, X)


a = 1.5
xlim = np.power(10, .3)
N = 100000
norm = (1 - CDF(xlim, a)) * gamma(a)

randomU = CDF(xlim, a) + (1 - CDF(xlim, a)) * np.random.rand(N)
xi = invCDF(randomU, a)
bins = np.power(10, np.arange(0, 2.01, .1))
hist, bins = np.histogram(xi, bins=bins)
hist = hist / N
hist = hist * norm / np.diff(bins)
mids = np.power(10, .5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))

fig = pp.figure(3)
sp1 = fig.add_subplot(1, 1, 1)
xdata = np.logspace(0, 1, 200)
pp.plot(np.log10(xdata), np.log10(f(xdata, a)), '-k', lw=1.5)
pp.axvline(x=np.log10(xlim), ls='dotted', color='black')
pp.xlabel(r'$\log_{10}x$')
pp.ylabel(r'$\log_{10}f(x)$')
pp.errorbar(np.log10(mids), np.log10(hist), marker='.', ls='None', xerr=.05, ecolor='black', mec='black', mfc='black')

pp.savefig(pdffile, format='pdf')


#-----------schechter function, dimensionless, alpha


del a


def f(x, alpha):
    return np.power(x, alpha) * np.exp(-x)


def CDF(x, alpha):
    return gammainc(alpha + 1, x)


def invCDF(X, alpha):
    return gammaincinv(alpha + 1, X)


alpha = -.5
xlim = np.power(10, -1.)
N = 100000
norm = (1 - CDF(xlim, alpha)) * gamma(alpha + 1)

randomU = CDF(xlim, alpha) + (1 - CDF(xlim, alpha)) * np.random.rand(N)
xi = invCDF(randomU, alpha)
bins = np.power(10, np.arange(-2., 2.01, .1))
hist, bins = np.histogram(xi, bins=bins)
hist = hist / N / np.diff(bins)
mids = np.power(10, .5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))

fig = pp.figure(5)
sp1 = fig.add_subplot(1, 1, 1)
xdata = np.logspace(-1.5, 1.5, 200)
pp.plot(np.log10(xdata), np.log10(f(xdata, alpha) / norm), '-k', lw=1.5)
pp.axvline(x=np.log10(xlim), ls='dotted', color='black')
pp.xlabel(r'$\log_{10}x$')
pp.ylabel(r'$\log_{10}f(x)$')
pp.errorbar(np.log10(mids), np.log10(hist), marker='.', ls='None', xerr=.05, ecolor='black', mec='black', mfc='black')

pp.savefig(pdffile, format='pdf')


#-----------schechter function, alpha, Mstar


def f(M, theta):
    alpha, Mstar = theta
    return np.power(Mstar, -1.) * np.power(M / Mstar, alpha) * np.exp(-M / Mstar)


def CDF(M, theta):
    alpha, Mstar = theta
    return gammainc(alpha + 1, M / Mstar)


def invCDF(X, theta):
    alpha, Mstar = theta
    return Mstar * gammaincinv(alpha + 1, X)


theta = -.5, np.power(10, 10)
alpha, Mstar = theta
Mlim = np.power(10, 8.5)
N = 1000
norm = (1 - CDF(Mlim, theta)) * gamma(alpha + 1)

randomU = CDF(Mlim, theta) + (1 - CDF(Mlim, theta)) * np.random.rand(N)
Mi = invCDF(randomU, theta)
bins = np.power(10, np.arange(8, 11.01, .25))
hist, bins = np.histogram(Mi, bins=bins)
hist = hist / N / np.diff(bins)
mids = np.power(10, .5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))

fig = pp.figure(6)
sp1 = fig.add_subplot(1, 1, 1)
xdata = np.logspace(8, 11, 200)
pp.plot(np.log10(xdata), np.log10(f(xdata, theta) / norm), '-k', lw=1.5)
pp.axvline(x=np.log10(Mlim), ls='dotted', color='black')
pp.xlabel(r'$\log_{10}M$')
pp.ylabel(r'$\log_{10}f(M)$')
pp.errorbar(np.log10(mids), np.log10(hist), marker='.', ls='None', xerr=.125, ecolor='black', mec='black', mfc='black')


def logPrior(theta):
    alpha, Mstar = theta
    if (Mstar > np.log10(Mlim)) and (alpha > -1):
        return 0.0
    else:
        return -np.inf


def logLikelihood(theta, Mi):
    alpha, Mstar = theta
    return np.sum(
        alpha * np.log(Mi / np.power(10, Mstar)) - 
        Mi / np.power(10, Mstar) - 
        np.log(np.power(10, Mstar)) - 
        np.log(gammaincc(alpha + 1, Mlim / np.power(10, Mstar)) * gamma(alpha + 1))
    )


def logProbability(theta, Mi):
    lp = logPrior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + logLikelihood(theta, Mi)


ndim, nwalkers = 2, 100
pos = [np.array([0, 10]) + np.array([.01, .01]) * (np.random.rand(2) - .5) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=(Mi, ))
sampler.run_mcmc(pos, 2000)
theta_ml = sampler.chain[np.unravel_index(sampler.lnprobability.argmax(), sampler.lnprobability.shape)]
samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

print(norm)
pp.plot(np.log10(xdata), np.log10(f(xdata, (theta_ml[0], np.power(10, theta_ml[1]))) / norm), '-r', lw=1.)

pp.savefig(pdffile, format='pdf')

cornerfig = pp.figure(7)
for spn in range(1, 5):
    cornerfig.add_subplot(2, 2, spn)
corner.corner(samples, labels=[r'$\alpha$', r'$M_\star\,[{\rm M}_\odot]$'], fig=cornerfig)

pp.savefig(pdffile, format='pdf')


pdffile.close()
