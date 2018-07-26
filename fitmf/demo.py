import numpy as np
import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages
from fitmf import schechter
from fitmf.sty import STYFitter

pdffile = PdfPages('demo.pdf')

theta = -.5, np.power(10, 10)
alpha, Mstar = theta
Mlim = np.power(10, 8.5)
input_norm = schechter.normalized_integral(theta, Mlim)
N = 1000

Mi = schechter.simulate(N, theta, Mlim)

fitter = STYFitter(Mi, Mlim)
fitter.fit(theta) #used input as guess
theta_ml = (
    fitter.results['theta_ml'][0],
    np.power(10, fitter.results['theta_ml'][1])
)
norm = schechter.normalized_integral(theta_ml, Mlim)

bins = np.power(10, np.arange(8, 11.01, .25))
hist, bins = np.histogram(Mi, bins=bins)
hist = hist / N / np.diff(bins)
mids = np.power(10, .5 * (np.log10(bins[1:]) + np.log10(bins[:-1])))

fig = pp.figure(1)
sp1 = fig.add_subplot(1, 1, 1)
sp1.set_xlabel(r'$\log_{10}M$')
sp1.set_ylabel(r'$\log_{10}\phi(M)?$')
xdata = np.logspace(8, 11, 200)
sp1.plot(
    np.log10(xdata),
    np.log10(schechter.function(xdata, theta) / input_norm),
    '-k',
    lw=1.5
)
sp1.axvline(x=np.log10(Mlim), ls='dotted', color='black')
sp1.errorbar(
    np.log10(mids),
    np.log10(hist),
    marker='.',
    ls='None',
    xerr=.125,
    ecolor='black',
    mec='black',
    mfc='black'
)
sp1.plot(
    np.log10(xdata),
    np.log10(schechter.function(xdata, theta_ml) / norm),
    ls='solid',
    marker='None',
    color='red',
    lw=1.
)
pp.savefig(pdffile, format='pdf')

fitter.cornerfig(save=pdffile, fignum=2)

pdffile.close()
