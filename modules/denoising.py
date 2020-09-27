"""
ref Book : Machine Learning for Asset Managers (Ch2.Denosing and Detoning)
codes for Denoising and Detoning

Author : Eugene Kim (brownian@kakao.com)
"""

# Import general libs
import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import minimize

# Import project defined libs
from .general_methods import cov2corr, corr2cov
from .data_generator import mpPDF


def getPCA(matrix:np.array):
    """
    Get Principal Components Analysis results
    Snippet 2.2 Testing the Marcdnko-Pastur Theorem
    Args:
        matrix:

    Returns:

    """
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc

    eVal = eVal[indices]
    eVec = eVec[:, indices]

    eVal = np.diagflat(eVal)

    return eVal, eVec


def fitKDE(obs, bWidth=0.25, kernel='gaussian', x=None):
    """
    Fit observation with Kernel Density Estimation Methods
    Snippet 2.2 Testing the Marcdnko-Pastur Theorem
    Args:
        obs:
        bWidth:
        kernel:
        x:

    Returns:

    """
    # Fit kernel to a series of obs, and derive the probability of obs
    # x : the array of values on which the fit KDE will be evaluated

    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)

    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())

    return pdf


def errPDFs(var, eVal, q, bWidth, pts=1000):
    """
    Fit error Theoretical pdf
    Snippet 2.4 Fitting the Marcenko-Pastur PDF
    Args:
        var:
        eVal:
        q:
        bWidth:
        pts:

    Returns:

    """


    # prevent var values in list format
    if hasattr(var, "__len__"):
        if len(var) == 1:
            var = var[0]
        else:
            raise ValueError("var  must be scalar")

    pdf0 = mpPDF(var, q, pts)
    # Empirical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)
    sse = np.sum((pdf1 - pdf0) ** 2)

    return sse


def findMaxEval(eVal, q, bWidth):
    """
    Find Max random eVal by fitting Marcenko's distribution

    Snippet 2.4 Fitting the Marcenko-Pastur PDF
    Args:
        eVal:
        q:
        bWidth:

    Returns:

    """

    out = minimize(fun=lambda *x: errPDFs(*x), x0=.5,  # first arg : var
                   args=(eVal, q, bWidth), bounds=((1e-5, 1 - 1e-5),))

    if out['success']:
        var = out['x'][0]
    else:
        var = 1

    eMax = var * (1 + (1.0 / q) ** 0.5) ** 2

    return eMax, var


def denoisedCorr(eVal, eVec, nFacts, shrink:bool=False, alpha:float=0.0):
    """
    Remove noise from corr by fixing random eigenvalues
    Snippet 2.5 Denoising by constant residual eigenvalue
    Args:
        eVal:
        eVec:
        nFacts:
        shrink: Choose shrinkage method(True) or not
        alpha: when use shrinkage, set alpha (0.0~1.0)

    Returns:

    """
    if shrink :
        # Remove noisse from corr through targeted shrinkage
        eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
        eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]

        corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
        corr1 = np.dot(eVecR, eValR).dot(eVecR.T)

        corr_ret = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    else :

        eVal_ = np.diag(eVal).copy()
        eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
        eVal_ = np.diag(eVal_)

        corr1 = np.dot(eVec, eVal_).dot(eVec.T)
        corr_ret = cov2corr(corr1)

    return corr_ret


def deNoiseCov(cov0, q, bWidth):
    """
    TODO : comments
    Snippet 2.9 Denoising of the Empirical Covariance Matrix
    Args:
        cov0:
        q:
        bWidth:

    Returns:

    """
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)

    # Finding noise starting point
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)

    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)

    cov1 = corr2cov(corr1, np.diag(cov0) ** 0.5)

    return cov1


def optPort(cov, mu=None):
    """
    TODO : comments
    Snippet 2.10 Denoising of the Emprical Covariance Matrix
    Args:
        cov:
        mu:

    Returns:

    """


    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones

    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)

    return w