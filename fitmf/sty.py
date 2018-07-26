import numpy as np
import matplotlib.pyplot as pp
from scipy.special import gammaincc, gamma
import astropy.units as U
import emcee
import corner


def _logPrior(theta, Mlim=0):
    alpha, Mstar = theta
    if (Mstar > np.log10(Mlim)) and (alpha > -1):
        return 0.0
    else:
        return -np.inf


def _logLikelihood(theta, Mi, Mlim=0):
    alpha, Mstar = theta
    return np.sum(
        alpha * np.log(Mi / np.power(10, Mstar)) - 
        Mi / np.power(10, Mstar) - 
        np.log(np.power(10, Mstar)) - 
        np.log(
            gammaincc(alpha + 1, Mlim / np.power(10, Mstar)) *
            gamma(alpha + 1)
        )
    )


def _logProbability(theta, Mi, Mlim=0):
    lp = _logPrior(theta, Mlim=Mlim)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + _logLikelihood(theta, Mi, Mlim=Mlim)


class STYFitter(object):

    def __init__(self, Mi, Mlim, nwalkers=100):
        try:
            self.Mi = Mi.to(U.Msun).value
        except AttributeError:
            self.Mi = Mi
        try:
            self.Mlim = Mlim.to(U.Msun).value
        except AttributeError:
            self.Mlim = Mlim
        self.nwalkers = nwalkers
        self.ndim = 2
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            _logProbability,
            args=(self.Mi, ),
            kwargs={'Mlim': self.Mlim}
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
