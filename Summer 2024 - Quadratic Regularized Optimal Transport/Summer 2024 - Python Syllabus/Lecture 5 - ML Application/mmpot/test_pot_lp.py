import numpy as np
import matplotlib.pyplot as plt
import ot
from ot.datasets import make_1D_gauss as gauss
import os
from pot_solvers.lp import lp
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.seterr(divide='ignore')
np.random.seed(100)
# bin positions
m, n = 150, 100
x1 = np.arange(m, dtype=np.float64)
x2 = np.arange(n, dtype=np.float64)
min_, max_ = min(x1.max(), x2.max()), max(x1.max(), x2.max())
# x1 = x1 * max_ / x1.max()
# x2 = x2 * max_ / x2.max()
# Gaussian mixtures
r = gauss(m, m=1 * m / 5, s=m / 20)  # m = mean, s = std
r += gauss(m, m=m / 2, s=m / 10)
r += 3 * gauss(m, m=4 * m / 5, s=m / 6)
# r /= a.sum()
c = gauss(n, m=3 * n / 5, s=n / 10)
c += 2 * gauss(n, m=35 * n / 100, s=n / 5)
# c /= b.sum()
# loss matrix
C = ot.dist(x1.reshape((m, 1)), x2.reshape((n, 1)))
C /= C.max()
s = 0.9 * min(r.sum(), c.sum())

source_hist = r
target_hist = c

fig = plt.figure(1, figsize=(6.4, 3))
plt.plot(x1, r, 'b', label='Source distribution (mass = {:.1f})'.format(r.sum()))
plt.plot(x2, c, 'r', label='Target distribution (mass = {:.1f})'.format(c.sum()))
plt.legend()
plt.title("Source and target distributions. Transported mass = {:.1f}".format(s))
plt.show()
plt.close(fig)

X_lp = lp(source_hist, target_hist, C, s, tol=1e-24, verbose=False)
f_star = np.sum(X_lp * C)

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
cmap = sns.color_palette("rocket_r", as_cmap=True)
sns.heatmap(X_lp, ax=ax, cmap=cmap)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"POT Matrix: LP", size=15)
fig.tight_layout()
plt.show()
plt.close(fig)
