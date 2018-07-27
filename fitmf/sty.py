import numpy as np
import matplotlib.pyplot as pp
from scipy.special import gammaincc, gamma
from scipy.optimize import fsolve
from scipy.stats import binned_statistic_2d
import emcee
import corner


def _logPrior(theta, Mlimi):
    alpha, Mstar = theta
    if (Mstar > Mlimi).any() and (alpha > -1):
        return 0.0
    else:
        return -np.inf


def _logLikelihood(theta, Mi, Mlimi, cMbins, cdbins, ci, cdi):
    alpha, Mstar = theta
    denominator = np.sum(
        cdi * gamma(alpha + 1) *
        (
            gammaincc(alpha + 1, np.power(10, cMbins[:-1] - Mstar)) -
            gammaincc(alpha + 1, np.power(10, cMbins[1:] - Mstar))
        )[:, np.newaxis],
        axis=0
    )
    return np.sum(
        ci +
        alpha * np.log(Mi / np.power(10, Mstar)) - 
        Mi / np.power(10, Mstar) - 
        np.log(np.power(10, Mstar)) - 
        np.log(denominator)
    )


def _logProbability(theta, Mi, Mlimi, cMbins, cdbins, ci, cdi):
    lp = _logPrior(theta, Mlimi)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + _logLikelihood(theta, Mi, Mlimi, cMbins, cdbins, ci, cdi)


class STYFitter(object):

    def __init__(self, Mi, di, cbundle, nwalkers=100):
        self.cMbins, self.cdbins = cbundle[1], cbundle[2]
        self.Mindices, self.dindices = \
            binned_statistic_2d(np.log10(Mi), di, None, statistic='count',
                                bins=[self.cMbins, self.cdbins],
                                expand_binnumbers=True)[3]
        self.cdi = self._cMatrix(*cbundle)[:, self.dindices - 1]
        self.ci = self.cdi[(self.Mindices - 1, np.arange(len(Mi)))]
        self.Mi = Mi
        self.di = di
        self.Mlimi = self._Mlim(self.di, cbundle[0])
        self.nwalkers = nwalkers
        self.ndim = 2
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            _logProbability,
            args=(self.Mi, self.Mlimi, self.cMbins, self.cdbins,
                  self.ci, self.cdi),
        )
        self.results = None
        return
    
    def fit(self, guess, niter=5500, burn=500):
        guess = np.array([guess[0], np.log10(guess[1])])
        pos = [guess + .01 * np.ones(2) * (np.random.rand(2) - .5)
               for i in range(self.nwalkers)]
        self.sampler.run_mcmc(pos, niter)
        self.results = {}
        self.results['theta_ml'] = self.sampler.chain[
            np.unravel_index(
                self.sampler.lnprobability.argmax(),
                self.sampler.lnprobability.shape
            )
        ]
        self.results['samples'] = self.sampler.chain[:, burn:, :]\
                                              .reshape((-1, self.ndim))
        return
    
    def cornerfig(self, save=None, save_format='pdf', fignum=999):
        if self.results is None:
            raise RuntimeError('Run fit before trying to create plot.')
        cornerfig = pp.figure(fignum)
        for spn in range(1, self.ndim ** 2 + 1):
            cornerfig.add_subplot(self.ndim, self.ndim, spn)
        corner.corner(
            self.results['samples'],
            labels=[r'$\alpha$', r'$M_\star\,[{\rm M}_\odot]$'],
            fig=cornerfig
        )
        if save is not None:
            pp.savefig(save, format=save_format)
        return
    
    def _cMatrix(self, cfunction, Mbins, dbins):
        dmids = .5 * (dbins[1:] + dbins[:-1])
        Mmids = .5 * (Mbins[1:] + Mbins[:-1])
        dgrid, Mgrid = np.meshgrid(dmids, Mmids)
        retval = cfunction(Mgrid, dgrid)
        retval[retval > 1] = 1
        retval[retval < 0] = 0
        return retval
    
    def _Mlim(self, d, c):
        return fsolve(lambda M: c(M, d), np.zeros(d.shape))
