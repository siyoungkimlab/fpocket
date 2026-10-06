"""Fit fpocket's pocket score for a representation, as fpocket's own was fitted.

fpocket ranks the pockets it finds by a linear score of pocket descriptors,
fitted on its training set of holo complexes (logistic regression on
binding vs other pockets for the current score; PLS before).  Here the same
model is fitted on the descriptors fpocket writes with
--write_score_descriptors (``evaluate.SCORE_FEATURES``), labelling a pocket
binding when it passes PPc.  Features are standardized for the fit and the
coefficients returned for the raw descriptors, in the order
``--score_coefficients`` takes them (intercept first).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import evaluate as E

NF = len(E.SCORE_FEATURES)
C = 1.0  # inverse L2 strength on standardized features


def tables(results):
    """Per structure: (features (n, NF), PPc labels (n,), MOc labels (n,)); None
    for a structure fpocket found no pocket in."""
    out = []
    for r in results:
        t = np.asarray(r.get("table") or [], float).reshape(-1, NF + 2)
        # a one-sphere pocket has no alpha-sphere density (NaN): count it as none
        X = np.nan_to_num(t[:, :NF], nan=0.0)
        out.append((X, t[:, NF].astype(bool), t[:, NF + 1].astype(bool)) if len(t) else None)
    return out


def fit(structs) -> np.ndarray:
    """Coefficients (intercept + NF) for the raw descriptors."""
    found = [s for s in structs if s is not None]
    if not found:  # no pocket anywhere: nothing to rank
        return np.r_[0.0, np.zeros(NF)]
    X = np.vstack([s[0] for s in found])
    y = np.concatenate([s[1] for s in found])
    if y.all() or not y.any():
        return np.r_[0.0, np.zeros(NF)]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    model.fit((X - mu) / sd, y)
    w = model.coef_[0] / sd
    return np.r_[model.intercept_[0] - (w * mu).sum(), w]


def ranks(structs, coeffs, label: int = 1) -> list:
    """Rank of the first correct pocket of each structure under ``coeffs`` (None
    if none is correct); ``label`` 1 for PPc, 2 for MOc."""
    out = []
    for s in structs:
        if s is None or not s[label].any():
            out.append(None)
            continue
        order = np.argsort(-(s[0] @ coeffs[1:] + coeffs[0]), kind="stable")
        out.append(int(np.flatnonzero(s[label][order])[0]) + 1)
    return out


def cv_ranks(structs, folds: int = 5) -> list:
    """Ranks with the score fitted on the other folds (structures held out whole)."""
    out = [None] * len(structs)
    groups = np.arange(len(structs))
    for train, test in GroupKFold(folds).split(groups, groups=groups):
        coeffs = fit([structs[i] for i in train])
        for i, r in zip(test, ranks([structs[i] for i in test], coeffs), strict=True):
            out[i] = r
    return out


def tops(rank_list, ks=(1, 3, 5, 10)) -> dict:
    n = len(rank_list)
    return {f"top{k}": sum(1 for r in rank_list if r is not None and r <= k) / n for k in ks}


def coefficient_flag(coeffs) -> str:
    return "--score_coefficients=" + ",".join(f"{c:.6g}" for c in coeffs)
