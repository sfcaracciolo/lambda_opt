from itertools import cycle
from src.lambda_opt import LCurve, UCurve, CRESO
from regularization_tools import Ridge, Tikhonov
from matplotlib import pyplot as plt
from geometric_plotter import Plotter
import numpy as np 

Plotter.set_export()

rng = np.random.default_rng(seed=3)

m, n = 50, 50
A = rng.random((m, n))
B = rng.random((n, n)) # np.eye(n)
y = rng.random((m, 1))
l_min, l_max = .01, 5.

settings = iter([
    dict(name='l', title='L-curve (log-log)', ylabel='$P $', xlabel='$R $'),
    dict(name='u', title='U-curve (log-log)', ylabel='$\\frac{1}{R^2}+\\frac{1}{P^2}$', xlabel='$\lambda$'),
    dict(name='c', title='CRESO', ylabel='$P^2 + 2\lambda^2 \\frac{d P^2}{d \lambda^2}$', xlabel='$\lambda $'),
])

regs = [
    Ridge(A),
    Tikhonov(A, B)
]

linestyle = cycle(['-k', '--k'])

for criteria in (LCurve, UCurve, CRESO):
    fig, ax = plt.subplots()
    pltset = next(settings)
    ax.set_title(pltset['title'])
    ax.set_ylabel(pltset['ylabel'])
    ax.set_xlabel(pltset['xlabel'])
    for reg in regs:
        reg.set_lambdas(l_min, l_max, 100)
        reg.solve(y)
        R = reg.compute_residuals()
        P = reg.compute_penalizations()
        ctr = criteria(P, R, reg.lambdas)
        opt = ctr.get_optimum()
        ax.plot(ctr.x, ctr.y, next(linestyle))
        ax.plot(ctr.x[opt], ctr.y[opt], '.k', markersize=10)
        # print(reg.lims)
    Plotter.save(folder='figs/', name=f"curve_{pltset['name']}")
Plotter.show()
