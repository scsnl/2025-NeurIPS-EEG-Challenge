import sys
import types
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import make_scorer

from typing import Optional, Union, Any, Dict, Iterable
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def closest(y, left, right):
    return np.abs(y - left) <= np.abs(y - right)

def towards_mean(y, left, right):
    return (left >= np.mean(y))

def away_mean(y, left, right):
    return (right <= np.mean(y))

class BiasRTRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, threshold, strategy = closest):
        self.estimator = estimator
        self.threshold = threshold
        self.strategy = strategy

    def fit(self, X, y):
        _y, _count = np.unique(y, return_counts = True)
        self.y_ = _y[_count>self.threshold]
        #self.estimator.fit(X,y)
        return self

    def predict(self, X):
        y = self.estimator.predict(X)
        i = np.searchsorted(self.y_, y)
        left, right = self.y_[np.clip(i-1, 0, len(self.y_)-1)], self.y_[np.clip(i, 0, len(self.y_)-1)]
        pick_left = np.abs(y - left) <= np.abs(y - right)
        return np.where(pick_left, left, right)

    def __sklearn_clone__(self):
        return BiasRTRegressor(self.estimator, self.threshold)

class MeanRegressor(RegressorMixin, BaseEstimator):
    def __init__(self):
        self._fitted = True

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.mean(axis=1)
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

class SumRegressor(RegressorMixin, BaseEstimator):
    def __init__(self):
        self._fitted = True

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.sum(axis=1)
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

class WeightedMeanRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, weights):
        self._fitted = True
        self.weights = weights

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.average(X, axis=1, weights = self.weights)
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

class SimpleStackingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators, final_estimator=None):
        self.estimators = estimators
        self.final_estimator = final_estimator or LinearRegression()

    def fit(self, X, y):
        try:
            check_is_fitted(self.final_estimator)
        except:
            Z = np.stack([e.predict(X) for _, e in self.estimators], axis = -1)
            self.final_estimator.fit(Z,y)
        return self

    def predict(self, X):
        _Z = [e.predict(X) for _, e in self.estimators]
        Z = np.stack(_Z, axis = -1)
        y = self.final_estimator.predict(Z)
        return y
    
    def __sklearn_clone__(self):
        return SimpleStackingRegressor(self.estimators, clone(self.final_estimator))

def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))/np.std(y_true)

neg_nrmse = make_scorer(nrmse, greater_is_better=False)

_Grid = Dict[str, Iterable[Any]]
_DEFAULT_GRID: _Grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [None, 3, 7],
    "max_leaf_nodes": [31, 63, 127],
    "l2_regularization": [0.0, 1e-4, 1e-3, 1e-2],
}

class HistGradientBoostingRegressorCV(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        param_grid: Optional[_Grid] = None,
        cv: Union[int, Any] = 5,
        scoring: Optional[Union[str, Any]] = "neg_mean_squared_error",
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
        error_score: Union[str, float] = np.nan,
        random_state: Optional[int] = None,
        **estimator_params,
    ):
        self.param_grid = _DEFAULT_GRID if param_grid is None else param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.random_state = random_state
        self.estimator_params = estimator_params

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        base = HistGradientBoostingRegressor(
            random_state=self.random_state, **self.estimator_params
        )
        gs = GridSearchCV(
            base,
            self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            refit=self.refit,
            verbose=self.verbose,
            error_score=self.error_score,
        )
        fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
        gs.fit(X, y, **fit_params)

        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = (
            gs.best_estimator_
            if self.refit
            else clone(base).set_params(**gs.best_params_).fit(X, y, **fit_params)
        )
        self.n_features_in_ = getattr(self.best_estimator_, "n_features_in_", None)
        self.feature_names_in_ = getattr(self.best_estimator_, "feature_names_in_", None)
        return self

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

def softmax(x, beta = 1.0):
    exps = np.exp(beta*x)
    return exps / np.sum(exps, axis=-1, keepdims=True)

class SimpleBoostingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators, max_step = 100):
        self.estimators = estimators
        self.n_estimators = len(estimators)
        self.ensemble = []
        self.max_step = max_step

    def fit(self, X, y):
        y_std = np.std(y)
        name,e = self.estimators[0]
        print(f'[BOOSTING] Building estimator {0:d}: {name}')
        self.ensemble.append(clone(e))
        self.ensemble[-1].fit(X,y)
        _y = self.ensemble[-1].predict(X)
        print(np.std(_y))
        res = y-_y
        print(f'Train NRMSE: {np.sqrt(np.mean(res**2))/y_std:.3e}')
        for i in range(1,self.max_step):
            name,e = self.estimators[i%self.n_estimators]
            print(f'[BOOSTING] Building estimator {i:d}: {name}')
            self.ensemble.append(clone(e))
            self.ensemble[-1].fit(X,res)
            __y = self.ensemble[-1].predict(X)
            print(np.std(__y))
            _y += __y
            res = y-_y
            print(f'Train NRMSE: {np.sqrt(np.mean(res**2))/y_std:.3e}')
        return self

    def predict(self, X):
        return np.stack([e.predict(X) for e in self.ensemble], axis = -1).sum(axis = -1)