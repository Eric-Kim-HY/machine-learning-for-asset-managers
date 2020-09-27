"""
ref Book : Machine Learning for Asset Managers
codes for general utilities for this projects (mostly fitting functions)

Author : Eugene Kim (brownian@kakao.com)
"""

# Import general libs
import pandas as pd
import numpy as np



def cov2corr(cov):
    """
    Derive the correlation matrix from a covariance matrix
    Snippet 2.3 Add signal to a random covariance matrix
    Args:
        cov:

    Returns:

    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    # Clipping to prevent numerical errors
    corr[corr < -1] = -1
    corr[corr > 1] = 1

    return corr

def corr2cov(corr, std):
    """
    Snippet 2.9 Denoising of the Empirical Covariance Matrix
    Args:
        corr:
        std:

    Returns:

    """
    cov = corr * np.outer(std,std)
    return cov