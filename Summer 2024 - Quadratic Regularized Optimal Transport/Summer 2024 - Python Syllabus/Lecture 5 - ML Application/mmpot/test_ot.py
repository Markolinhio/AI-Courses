import numpy as np
from matplotlib import pyplot as plt
from ot.datasets import make_1D_gauss as gauss
import os
import ot
from ot_solvers.sinkhorn import sinkhorn
from ot_solvers.apdagd import apdagd
from ot_solvers.apdamd import apdamd
from ot_solvers.pdasmd import pdasmd
import argparse
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.seterr(divide='ignore')
parser = argparse.ArgumentParser(description='Test the Sinkhorn algorithm.')
parser.add_argument("--n", default=100, type=int,
                    help="Number of dimensions (supports)")
parser.add_argument("--algorithm", default="sinkhorn", type=str,
                    help="Optimization algorithm",
                    choices=["sinkhorn", "apdagd", "apdamd", "pdasmd"])
parser.add_argument("--seed", default=100, type=int,
                    help="Random seed")
parser.add_argument("--epsilon", default=10 ** (-1.2), type=float,
                    help="Primal optimality tolerance")
parser.add_argument("--verbose", default=0, type=int,
                    help="Whether to print progress")
parser.add_argument("--print_every", default=10, type=int,
                    help="Print progress every this many iterations")
parser.add_argument("--plot_dist", default=0, type=int,
                    help="Plot the distributions")
parser.add_argument("--plot_otmatrix", default=0, type=int,
                    help="Plot the OT matrix")

args = parser.parse_args()
n = args.n
seed = args.seed
epsilon = args.epsilon
print_every = args.print_every

np.random.seed(seed)
# bin positions
x = np.arange(n, dtype=np.float64)
# Gaussian mixtures
a = gauss(n, m=1 * n / 5, s=n / 25)  # m = mean, s = std
a += gauss(n, m=n / 2, s=n / 10)
a += 3 * gauss(n, m=4 * n / 5, s=n / 6)
a /= a.sum()
b = gauss(n, m=4 * n / 5, s=n / 10)
b += gauss(n, m=35 * n / 100, s=n / 5)
b /= b.sum()
# loss matrix
C = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
C /= C.max()

plot_dist = bool(args.plot_dist)
if plot_dist:
    fig = plt.figure(1, figsize=(6.4, 3))
    plt.plot(x, a, 'b', label='Source distribution')
    plt.plot(x, b, 'r', label='Target distribution')
    plt.legend()
    plt.title("Source and target distributions")
    plt.show()
    plt.close(fig)

verbose = bool(args.verbose)
algorithm = args.algorithm
print(f"Optimizing using {algorithm}...")
if algorithm == "sinkhorn":
    OT_matrix, logs = sinkhorn(a, b, C,
                               num_iters=30000,
                               verbose=verbose,
                               print_every=print_every,
                               tol=epsilon)
elif algorithm == "apdagd":
    OT_matrix, logs = apdagd(a_dist=a, b_dist=b, C=C,
                             num_iters=30000,
                             verbose=verbose,
                             print_every=print_every,
                             tol=epsilon)
elif algorithm == "apdamd":
    OT_matrix, logs = apdamd(a_dist=a, b_dist=b, C=C,
                             num_iters=30000,
                             verbose=verbose,
                             print_every=print_every,
                             tol=epsilon)
elif algorithm == "pdasmd":
    OT_matrix, logs = pdasmd(a_dist=a, b_dist=b, C=C,
                             num_iters=30000,
                             verbose=verbose,
                             print_every=print_every,
                             tol=epsilon,
                             inner_iters=1)

plot_otmatrix = bool(args.plot_otmatrix)
if plot_otmatrix:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(OT_matrix, ax=ax, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"OT Matrix: {algorithm.upper()}", size=15)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
