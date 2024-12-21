# Multi-marginal Partial Optimal Transport

## Setting up a Python environment

We will be using Anaconda for setting up a virtual environment.

```sh
# Create a new environment
$ conda create --name mmpot python=3.9.12

# Activate the environment
$ conda activate mmpot

# Install packages and dependencies
(mmpot) $ pip install -r requirements.txt
```

The LP solver uses the GUROBI backend, which is a commercial software but is free for use for academic purposes. Check
the [GUROBI download page](https://www.gurobi.com/downloads/). You can also change the solver in `ot/lp.py`.