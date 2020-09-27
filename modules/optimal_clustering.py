"""
ref Book : Machine Learning for Asset Managers (Ch4.Optimal Clustering)
codes for Optimal Clustering

Author : Eugene Kim (brownian@kakao.com)
"""

# Import general libs
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Import project defined libs


def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    """
    TODO add comment
    Snippet 4.1 Base Clustering
    Args:
        corr0:
        maxNumClusters:
        n_init:

    Returns:

    """
    x = ((1 - corr0.fillna(0)) / 2) ** 0.5
    silh = pd.Series()  # Observations matrix

    for init in range(n_init):
        for i in range(2, maxNumClusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)

            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh = silh_
                kmeans = kmeans_

    # Reordering
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]  # reorder rows
    corr1 = corr1.iloc[:, newIdx]  # reorder columns

    clstrs = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() \
              for i in np.unique(kmeans.labels_)}  # cluster members
    silh = pd.Series(silh, index=x.index)

    return corr1, clstrs, silh


def makeNewOutputs(corr0, clstrs, clstrs2):
    """

    Snippet 4.2 Top-Level of Clustering
    Args:
        corr0:
        clstrs:
        clstrs2:

    Returns:

    """
    clstrsNew = {}

    # Allocate elements of cluster 1 and 2 to one dictionary (clstrsNew)
    for key_i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[key_i])

    for key_j in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[key_j])

    # need study
    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]

    # Rearrange the correlation matrix
    corrNew = corr0.loc[newIdx, newIdx]

    # Correlation to distance
    x = ((1 - corr0.fillna(0)) / 2.0) ** 0.5

    kmeans_labels = np.zeros(len(x.columns))
    print(kmeans_labels)

    # Get index using clstrsNew cluster and x's value
    for idx in clstrsNew.keys():
        idxs = [x.index.get_loc(k) for k in clstrsNew[idx]]

    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)

    return corrNew, clstrsNew, silhNew


def clusterKMeansTop(corr0, maxNumClusters=None, n_init=10):
    """
    TODO add comment
    Snippet 4.2 Top-Level of Clustering
    Args:
        corr0:
        maxNumClusters:
        n_init:

    Returns:

    """

    # If there's no maxNumClusters value, set its value lenth(corr) - 1
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1] - 1

    corr1, clstrs, silh = clusterKMeansBase(corr0,
                                            maxNumClusters=min(maxNumClusters, corr0.shape[1] - 1),
                                            n_init=n_init)

    clusterTstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()}

    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)

    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]

    if len(redoClusters) <= 1:
        return corr1, clstrs, silh

    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]

        corrTmp = corr0.loc[keysRedo, keysRedo]
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])

        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp,
                                                 maxNumClusters=min(maxNumClusters, corrTmp.shape[1] - 1),
                                                 n_init=n_init)

        # Make new outputs, if necessary

        corrNew, clstrsNew, silhNew = \
            makeNewOutputs(corr0,
                           {i: clstrs[i] for i in clstrs.keys() if i not in redoClusters},
                           clstrs2)

        newTstatMean = np.mean(
            [np.mean(silhNew[clstrsNew[i]]) / np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])

        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh

        else:
            return corrNew, clstrsNew, silhNew