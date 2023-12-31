from itertools import cycle
from src.lambda_opt import LCurve, UCurve, CRESO
from regularization_tools import Ridge, Tikhonov
from matplotlib import pyplot as plt
from geometric_plotter import Plotter
import numpy as np 

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

k = 0
for criteria in (LCurve, UCurve, CRESO):
    p = Plotter(_2d=True, ncols=1, nrows=1, figsize=(5,5))
    pltset = next(settings)
    p.axs.set_title(pltset['title'])
    p.axs.set_ylabel(pltset['ylabel'])
    p.axs.set_xlabel(pltset['xlabel'])
    for reg in regs:
        lambdas = reg.lambdaspace(l_min, l_max, 100)
        X = reg.solve(y, lambdas)
        R = reg.residual(X, y)
        P = reg.penalization(X)
        ctr = criteria(P, R, lambdas**2)
        opt = ctr.get_optimum()
        p.axs.plot(ctr.x, ctr.y, next(linestyle))
        p.axs.plot(ctr.x[opt], ctr.y[opt], '.k', markersize=10)
    p.save(folder='figs/', name=f"curve_{pltset['name']}")
Plotter.show()
