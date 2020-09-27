"""
ref Book : Machine Learning for Asset Managers (Ch4.Optimal Clustering)
codes for Optimal Clustering

Author : Eugene Kim (brownian@kakao.com)
"""

# Import general libs
import pandas as pd
import numpy as np

# Import project defined libs



def featImpMDI(fit, featNames):
    """
    Feature importance based on IS mean impurity reduction
    SNIPPET 6.2 Implementation of an Ensemble MDI Method
    Args:
        fit: fitted instance with sklearn tree methods (DecisionTreeClassifier, BaggingClassifier)
        featNames:

    Returns:

    """


    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}

    df0 = pd.DataFrame.from_dict(df0, orient='index')

    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1

    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -.5}, axis=1)  # CLT

    imp /= imp['mean'].sum()
    return imp


def featImpMDA(clf, X, y, n_splits=10):
    """
    feat importance based on OOS score reduction
    SNIPPET 6.3 Implementation of MDA

    Args:
        clf:
        X:
        y:
        n_splits:

    Returns:

    """


    from sklearn.metrics import log_loss
    from sklearn.model_selection._split import KFold
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0 = X.iloc[train, :], y.iloc[train]
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)  # the fit occurs here

        prob = fit.predict_proba(X1)  # prediction before shuffling
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # shuffle one column
            prob = fit.predict_proba(X1_)  # prediction after shuffling
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)

    imp = (-1 * scr1).add(scr0, axis=0)
    imp /= -1 * scr1
    imp = pd.concat({'mean': imp.mean(),
                     'std': imp.std() * imp.shape[0] ** -.5}, axis=1)  # CLT
    return imp


def groupMeanStd(df0, clstrs):
    """

    SNIPPET 6.4 Clustered MDI
    Args:
        df0:
        clstrs:

    Returns:

    """
    out = pd.DataFrame(columns=['mean', 'std'])

    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)
        out.loc['C_' + str(i), 'mean'] = df1.mean()
        out.loc['C_' + str(i), 'std'] = df1.std() * df1.shape[0] ** -.5

    return out


def featImpMDI_Clustered(fit, featNames, clstrs):
    """

    SNIPPET 6.4 Clustered MDI
    Args:
        fit:
        featNames:
        clstrs:

    Returns:

    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}

    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features = 1

    imp = groupMeanStd(df0, clstrs)
    imp /= imp['mean'].sum()

    return imp


def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10):
    """

    SNIPPET 6.5 Clustered MDA
    Args:
        clf:
        X:
        y:
        clstrs:
        n_splits:

    Returns:

    """
    from sklearn.metrics import log_loss
    from sklearn.model_selection._split import KFold

    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=clstrs.keys())

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0 = X.iloc[train, :], y.iloc[train]
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)
        prob = fit.predict_proba(X1)

        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)

        for j in scr1.columns:
            X1_ = X1.copy(deep=True)

            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values)  # shuffle clusters

            prob = fit.predict_proba(X1_)
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)

    imp = (-1 * scr1).add(scr0, axis=0)
    imp /= -1 * scr1
    imp = pd.concat({'mean': imp.mean(),
                     'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    imp.index = ['C_' + str(i) for i in imp.index]

    return imp