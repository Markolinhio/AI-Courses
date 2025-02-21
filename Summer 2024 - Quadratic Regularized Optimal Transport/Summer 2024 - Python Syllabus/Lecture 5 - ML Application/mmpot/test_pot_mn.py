import numpy as np
from matplotlib import pyplot as plt
from ot.datasets import make_1D_gauss as gauss
import os
from pot_solvers.apdagd_mn import apdagd_mn
from pot_solvers.sinkhorn_mn import sinkhorn_mn
import argparse
import seaborn as sns
import ot
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.seterr(divide='ignore')
parser = argparse.ArgumentParser(description='Test the Sinkhorn algorithm.')
parser.add_argument("--n", default=100, type=int,
                    help="Number of dimensions for source (supports)")
parser.add_argument("--m", default=100, type=int,
                    help="Number of dimensions for target (supports)")
parser.add_argument("--algorithm", default="sinkhorn", type=str,
                    help="Optimization algorithm",
                    choices=["sinkhorn", "apdagd", "apdamd", "dual_extrapolation"])
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
parser.add_argument("--save_logs", type=str, default="",
                    help="Save logs to this file")
parser.add_argument("--save_iterates", type=int, default=0,
                    help="Save iterates to logs")

args = parser.parse_args()
n = args.n
m = args.m
seed = args.seed
epsilon = args.epsilon
print_every = args.print_every

np.random.seed(seed)
# bin positions
x1 = np.arange(m, dtype=np.float64)
x2 = np.arange(n, dtype=np.float64)
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

plot_dist = bool(args.plot_dist)
if plot_dist:
    fig = plt.figure(1, figsize=(6.4, 3))
    plt.plot(x1, r, 'b', label='Source distribution (mass = {:.1f})'.format(r.sum()))
    plt.plot(x2, c, 'r', label='Target distribution (mass = {:.1f})'.format(c.sum()))
    plt.legend()
    plt.title("Source and target distributions. Transported mass = {:.1f}".format(s))
    plt.show()
    plt.close(fig)

verbose = bool(args.verbose)
algorithm = args.algorithm
print(f"Optimizing using {algorithm}...")
if algorithm == "apdagd":
    POT_matrix, logs = apdagd_mn(a_dist=r, b_dist=c, C=C, s=s,
                                 num_iters=1000,
                                 verbose=verbose,
                                 print_every=print_every,
                                 save_iterates=bool(args.save_iterates),
                                 tol=epsilon)
elif algorithm == "sinkhorn":
    POT_matrix, logs = sinkhorn_mn(a=r, b=c, C=C, s=s,
                                   num_iters=30000,
                                   verbose=verbose,
                                   print_every=print_every,
                                   save_iterates=bool(args.save_iterates),
                                   tol=epsilon)
if args.save_logs != "":
    with open(args.save_logs, "wb") as f:
        pickle.dump(logs, f)

plot_otmatrix = bool(args.plot_otmatrix)
if plot_otmatrix:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(POT_matrix, ax=ax, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"POT Matrix: {algorithm.upper()}", size=15)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
