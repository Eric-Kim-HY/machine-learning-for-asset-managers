"""
ref Book : Machine Learning for Asset Managers
codes for generating sample and simulation data (especially covariance matrix)

Author : Eugene Kim (brownian@kakao.com)
"""

# Import general libs
import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.utils import check_random_state

# Import project defined libs
from .general_methods import corr2cov, cov2corr


def mpPDF(var, q, pts) -> pd.Series:
    """
    Generates Marcenko-Pastur Pdf
    Snippet 2.1 The Marcenko_pastur PDF (in Machine Learning for Asset Managers)
    Args:
        var:
        q: T/N
        pts:

    Returns:

    """
    # Marcenko-Pastur pdf
    # q = T/N
    eMin = var * (1 - (1.0 / q) ** 0.5) ** 2
    eMax = var * (1 + (1.0 / q) ** 0.5) ** 2

    eVal = np.linspace(eMin, eMax, pts)

    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5

    if len(pdf.shape) > 1:
        pdf = pdf.squeeze()

    pdf = pd.Series(pdf, index=eVal)

    return pdf


def getRndCov(n_cols, n_facts):
    """
    Genarate Random Covariance Matrix with full rank
    Snippet 2.3 Add signal to a random covariance matrix
    Args:
        n_cols:
        n_facts:

    Returns:

    """
    w = np.random.normal(size=(n_cols, n_facts))

    # Random cov matrix, however not full rank
    cov = np.dot(w, w.T)

    # Make full rank cov
    cov += np.diag(np.random.uniform(size=n_cols))

    return cov


def formBlockMatrix(nBlocks, bSize, bCorr):
    """
    TODO : comments
    Snippet 2.7 Generating a Block-Diagonal Covariance Matrix and a Vector of Means
    Args:
        nBlocks:
        bSize:
        bCorr:

    Returns:

    """
    # make square matrix (bSize * bSize) (all elements' value is bCorr)
    block = np.ones((bSize, bSize)) * bCorr

    # change diagonal elements to 1
    block[range(bSize), range(bSize)] = 1

    # Make block diagonal matrix with size (bSize*nBlocks by bSize*nBlocks)
    corr = block_diag(*([block] * nBlocks))

    return corr


def formTrueMatrix(nBlocks, bSize, bCorr):
    """
    TODO : comments
    Snippet 2.7 Generating a Block-Diagonal Covariance Matrix and a Vector of Means
    Args:
        nBlocks:
        bSize:
        bCorr:

    Returns:

    """
    # make BlockMatrix in DataFrame Format
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)

    # make columns index to list and shuffle
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)

    # corr matrix shuffled along column & index
    corr0 = corr0[cols].loc[cols].copy(deep=True)

    # final calculations
    std0 = np.random.uniform(low=0.05, high=0.2, size=corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(loc=std0, scale=std0, size=cov0.shape[0]).reshape(-1, 1)

    return mu0, cov0


def simCovMu(mu0, cov0, nObs, shrink=False):
    """
    TODO : comments
    Snippet 2.8 Generating the Empirical Covariance Matrix
    Args:
        mu0:
        cov0:
        nObs:
        shrink:

    Returns:

    """
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)

    mu1 = x.mean(axis=0).reshape(-1, 1)

    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=False)

    return mu1, cov1


def getCovSub(nObs, nCols, sigma, random_state=None):
    """
    TODO add comment
    Snippet 4.3 Random Block Correlation Matrix Creation
    Args:
        nObs:
        nCols:
        sigma:
        random_state:

    Returns:

    """
    # Sub correl matrix
    rng = check_random_state(random_state)

    if nCols == 1:
        return np.ones((1, 1))

    ar0 = rng.normal(size=(nObs, 1))
    ar0 = np.repeat(ar0, nCols, axis=1)
    ar0 += rng.normal(scale=sigma, size=ar0.shape)
    ar0 = np.cov(ar0, rowvar=False)

    return ar0


def getRndBlockCov(nCols, nBlocks, minBlockSize=1, sigma=1.0, random_state=None):
    """
    Generate a block random correlation matrix
    Snippet 4.3 Random Block Correlation Matrix Creation
    Args:
        nCols:
        nBlocks:
        minBlockSize:
        sigma:
        random_state:

    Returns:

    """
    rng = check_random_state(random_state)
    parts = rng.choice(range(1, nCols - (minBlockSize - 1) * nBlocks), nBlocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, nCols - (minBlockSize - 1) * nBlocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + minBlockSize

    cov = None

    for nCols_ in parts:
        cov_ = getCovSub(int(max(nCols_ * (nCols_ + 1) / 2.0, 100)),
                         nCols_,
                         sigma,
                         random_state=rng)

        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)

    return cov


def randomBlockCorr(nCols, nBlocks, random_state=None, minBlockSize=1):
    """
    TODO add comment
    Snippet 4.3 Random Block Correlation Matrix Creation
    Args:
        nCols:
        nBlocks:
        random_state:
        minBlockSize:

    Returns:

    """
    # Form block corr
    rng = check_random_state(random_state)

    cov0 = getRndBlockCov(nCols, nBlocks,
                          minBlockSize=minBlockSize,
                          sigma=0.5,
                          random_state=rng)

    # Add Noise
    cov1 = getRndBlockCov(nCols, 1,
                          minBlockSize=minBlockSize,
                          sigma=1.0,
                          random_state=rng)

    cov0 += cov1

    corr0 = cov2corr(cov0)
    corr0 = pd.DataFrame(corr0)

    return corr0