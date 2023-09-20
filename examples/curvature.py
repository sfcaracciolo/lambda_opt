from src.lambda_opt import LCurve
from regularization_tools import Tikhonov
from matplotlib import pyplot as plt
from geometric_plotter import Plotter
import numpy as np 
from pathlib import Path

Plotter.set_export()

rng = np.random.default_rng(seed=3)

m, n = 50, 50
A = rng.random((m, n))
B = rng.random((n, n)) # np.eye(n)
y = rng.random((m, 1))
l_min, l_max = .01, .3


reg = Tikhonov(A, B)


reg.set_lambdas(l_min, l_max, 100)
reg.solve(y)
R = reg.compute_residuals()
P = reg.compute_penalizations()
ctr = LCurve(P, R, reg.lambdas)
opt = ctr.get_optimum()
(cx, cy), r = LCurve.osculating_circle(ctr.x, ctr.y, ctr.lambdas)
k = np.reciprocal(r)

patches = [
    plt.Circle((cx[opt], cy[opt]), r[opt], fill=False, color='k', linewidth=.5),
    
    plt.Circle((cx[opt-25], cy[opt-25]), r[opt-25], fill=False, color='k', linestyle='--', linewidth=.2),
]

fig, ax1 = plt.subplots(figsize=(8,8))
ax1.set_title('L-curve (log-log)')
ax1.set_ylabel('$P $')
ax1.set_xlabel('$R $')

ixs = (0,-10,-15)

# segments
[ax1.plot([cx[opt+i], ctr.x[opt+i]], [cy[opt+i], ctr.y[opt+i]], 'ko-', linewidth=.5) for i in ixs]

ax1.plot(ctr.x, ctr.y, '-k')

# opt
ax1.plot(ctr.x[opt], ctr.y[opt], '.k', markersize=20)
ax1.axvline(ctr.x[opt], linewidth=.1, linestyle='--' , color='gray')
ax1.axhline(ctr.y[opt], linewidth=.1, linestyle='--' , color='gray')

# circles
[ax1.add_patch(plt.Circle((cx[opt+i], cy[opt+i]), r[opt+i], fill=False, color='k', linestyle='--', linewidth=.2),) for i in ixs]

ax1.set_xlim((-.7, -.3))
ax1.set_ylim((.6, 1.))

ax2 = ax1.twinx()
ax2.plot(ctr.x, k, '--k', linewidth=1)
ax2.set_ylabel('$\kappa$')

Plotter.save(folder='figs/', name=Path(__file__).stem)
Plotter.show()
