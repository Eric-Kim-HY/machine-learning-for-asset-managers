"""
ref Book : Machine Learning for Asset Managers (Ch3. Distance Metrics)
codes for Mutual Information, Variation of Information


Author : Eugene Kim (brownian@kakao.com)
"""

# Import general libs
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

# Import project defined libs


def num_bins(nObs, corr=None):
    """
    TODO : add comment
    Snippet 3.3 Varation of Information on Discretized Continuous Random Variables
    Args:
        nObs:
        corr:

    Returns:

    """
    # Optimal number of bins for discretization
    if corr is None:  # univariate case
        z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs ** 2) ** 0.5) ** (1 / 3)
        b = round(z / 6 + 2 / (3 * z) + 1 / 3)
    else:  # bivariate case
        b = round(2 ** (-0.5) * (1 + (1 + 24 * nObs / (1 - corr ** 2)) ** 0.5) ** 0.5)

    return int(b)


def var_info(x, y, norm=False):
    """
    Variation of Information
    Snippet 3.3 Varation of Information on Discretized Continuous Random Variables
    Args:
        x:
        y:
        norm:

    Returns:

    """
    bXY = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)

    hX = ss.entropy(np.histogram(x, bXY)[0])  # Marginal
    hY = ss.entropy(np.histogram(y, bXY)[0])  # Marginal
    vXY = hX + hY - 2 * iXY  # Variation of Information

    if norm:
        hXY = hX + hY - iXY  # Joint
        vXY /= hXY  # Normalized Variation of Information

    return vXY


def mutualInfo(x, y, norm=False):
    """
    TODO add comment
    Snippet 3.4 Correlation and Normalized Mutual Information of Two Independent Gaussian Random Variables
    Args:
        x:
        y:
        norm:

    Returns:

    """
    # Mutual Information
    bXY = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)

    if norm:
        hX = ss.entropy(np.histogram(x, bXY)[0])  # Marginal
        hY = ss.entropy(np.histogram(y, bXY)[0])  # Marginal
        iXY /= min(hX, hY)  # Normalized mutual information
    return iXY