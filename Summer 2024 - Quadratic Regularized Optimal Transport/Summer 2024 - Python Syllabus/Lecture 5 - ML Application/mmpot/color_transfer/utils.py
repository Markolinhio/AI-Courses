import numpy as np
import matplotlib.image as img
from scipy.io import savemat


def rgb2luv(I):
    """
    Convert an image from rgb to luv color space, where
    l = r + g + b
    u = r / l
    v = g / l.
    This reduces the dimensionality of the color space from 3 to 2.
    """
    J = np.zeros(I.shape)
    J[:, :, 0] = np.sum(I, axis=2)
    J[:, :, 1] = I[:, :, 0] / (1e-14 + J[:, :, 0])
    J[:, :, 2] = I[:, :, 1] / (1e-14 + J[:, :, 0])
    return J


def luv2rgb(J):
    """
    Convert an image from luv to rgb color space.
    """
    I = np.zeros(J.shape)
    I[:, :, 0] = (J[:, :, 0] + 1e-14) * J[:, :, 1]
    I[:, :, 1] = (J[:, :, 0] + 1e-14) * J[:, :, 2]
    I[:, :, 2] = (J[:, :, 0] + 1e-14) * (1.0 - J[:, :, 1] - J[:, :, 2])
    return I


def make_2d_histogram(X, B):
    H = np.zeros((B, B))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = X[i, j, 1]
            g = X[i, j, 2]

            k = int(np.clip(B * r, 0, B - 1))
            l = int(np.clip(B * g, 0, B - 1))

            H[k, l] += 1.0

    H /= X.shape[0] * X.shape[1]
    return H


def make_3d_histogram(X, B):
    H = np.zeros((B, B, B))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = int(np.clip(B * X[i, j, 0], 0, B - 1))
            g = int(np.clip(B * X[i, j, 1], 0, B - 1))
            b = int(np.clip(B * X[i, j, 2], 0, B - 1))
            H[r, g, b] += 1.0
    H /= X.shape[0] * X.shape[1]
    return H


def apply_transport(T, Y, luvY=None):
    """
    Apply transport plan T to image Y.
    Args:
        T: Transport matrix of shape (B**2, B**2)
        Y: Original image in rgb
        luvY: Original image in luv
    """
    if luvY is None:
        luvY = rgb2luv(Y)
    B = int(np.sqrt(np.shape(T)[0]))
    om = np.zeros((B, B, 3))
    for i in range(B):
        for j in range(B - i):
            som = 1e-14
            for k in range(B):
                for l in range(B - k):
                    om[i, j, 0] += k * T[B * i + j, B * k + l]
                    om[i, j, 1] += l * T[B * i + j, B * k + l]
                    om[i, j, 2] += (B - 1 - k - l) * T[B * i + j, B * k + l]
                    som += T[B * i + j, B * k + l]
            om[i, j, 0] /= som
            om[i, j, 1] /= som
            om[i, j, 2] /= som

    cY = np.zeros(Y.shape)
    nY = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            r = luvY[i, j, 1]
            g = luvY[i, j, 2]
            b = 1 - r - g

            k = int(np.clip(B * r, 0.0, B - 1.0))
            l = int(np.clip(B * g, 0.0, B - 1.0))

            cY[i, j, :] = (luvY[i, j, 0] + 1e-14) * np.array([r, g, b])
            nY[i, j, :] = (luvY[i, j, 0] + 1e-14) * om[k, l, :] / (B - 1.0)

    return cY, nY


def setup_color_transfer(source_path, target_path, output_path=None, B=32, normalize_hist=True):
    """
    Setup a color transfer task.
    Args:
        source_path: Path to source image
        target_path: Path to target image
        output_path: Path to store the
        B: Number of bins for histograms
        normalize_hist: WHether to normalize the histograms to they sum to 1

    Returns:
        A dictionary with the following keys:
            source_rgb: Source image in rgb of shape (w, h, 3)
            target_rgb: Target image in rgb of shape (w, h, 3)
            source_luv: Source image in luv of shape (w, h, 3)
            target_luv: Target image in luv of shape (w, h, 3)
            source_hist: Source histogram of shape (B**2,)
            target_hist: Target histogram of shape (B**2,)
            cost_matrix: Cost matrix of shape (B**2, B**2)
    """

    # Read source image
    source_rgb = img.imread(source_path)
    source_rgb = source_rgb / 255
    source_luv = rgb2luv(source_rgb)

    # Read target image
    target_rgb = img.imread(target_path)
    target_rgb = target_rgb / 255
    target_luv = rgb2luv(target_rgb)

    # Create color histograms
    source_hist, _, _ = np.histogram2d(x=source_luv.reshape((-1, 3))[:, 1].flatten(),
                                       y=source_luv.reshape((-1, 3))[:, 2].flatten(),
                                       bins=B, range=[[0, 1], [0, 1]])
    target_hist, _, _ = np.histogram2d(x=target_luv.reshape((-1, 3))[:, 1].flatten(),
                                       y=target_luv.reshape((-1, 3))[:, 2].flatten(),
                                       bins=B, range=[[0, 1], [0, 1]])

    # Smooth histograms so no bin is empty
    eps = 1e-7
    source_hist += eps
    target_hist += eps

    # Normalize histograms
    if normalize_hist:
        source_hist /= source_hist.sum()
        target_hist /= target_hist.sum()

    # Create cost matrix
    # C = 4 * (B - 1.0) ** 2 * np.ones((B * B, B * B))
    # for i in range(B):
    #     for j in range(B - i):
    #         for k in range(B):
    #             for l in range(B - k):
    #                 C[B * i + j, B * k + l] = (i - k) ** 2 + (j - l) ** 2
    # C /= (B - 1.0) ** 2

    C = np.zeros((B * B, B * B), dtype=float)
    for i in range(B):
        for j in range(B):
            for k in range(B):
                for l in range(B):
                    row = i * B + j
                    col = k * B + l
                    C[row, col] = (i - k) ** 2 + (j - l) ** 2
    C = C / np.max(C)

    source_hist = source_hist.flatten()
    if len(source_hist.shape) != 1:
        source_hist = source_hist[0]

    target_hist = target_hist.flatten()
    if len(target_hist.shape) != 1:
        target_hist = target_hist[0]

    result = {
        'source_rgb': source_rgb,
        'source_luv': source_luv,
        'target_rgb': target_rgb,
        'target_luv': target_luv,
        'source_hist': source_hist,
        'target_hist': target_hist,
        'cost_matrix': C,
    }

    # Save result if output path is given
    if output_path is not None:
        savemat(output_path, result)

    return result
