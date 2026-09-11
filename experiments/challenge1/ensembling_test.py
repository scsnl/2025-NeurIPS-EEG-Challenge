
import sys
from os import getenv
import argparse
from pathlib import Path
from joblib import load, dump
import numpy as np

from sklearn.linear_model import RidgeCV
from eegchallenge.sklearn_models import *
from eegchallenge import config

def load_estimators(paths):
    return [(Path(path).stem, load(path)) for path in paths]

def main(args):
    print('Loading data')
    data_path = config.arrays_dir(1)
    X, y = np.load(f'{data_path}/X_test.npy'), np.load(f'{data_path}/y_test.npy')[:,0]

    print('Loading ensemble')
    ensemble_paths = sorted(args.model, key=lambda s: (len(s.stem), s.stem))
    if not args.k is None:
        ensemble_paths = ensemble_paths[:args.k]
    estimators = load_estimators(ensemble_paths)
    
    model_name = '_'.join([e[0] for e in estimators]) if args.name is None else args.name
    print(args.regressor)
    if args.regressor == 'mean':
        model = SimpleStackingRegressor(estimators, MeanRegressor())
        model_name = 'mean_'+model_name
    elif args.regressor[:5] == 'wmean':
        beta = float(args.regressor[5:]) if len (args.regressor[5:]) > 0 else 1.0
        errors = np.array([nrmse(y, e[1].predict(X)) for e in estimators])
        r2 = 1-errors**2
        print(r2)
        weights = softmax(r2, beta)
        print(weights)
        model = SimpleStackingRegressor(estimators, WeightedMeanRegressor(weights = weights))
        model_name = f'wmean_beta{beta:.2e}_'+model_name
    elif args.regressor == 'ridge':
        model = SimpleStackingRegressor(estimators, RidgeCV(alphas = [0.0, 0.1, 1.0, 10.0], scoring = neg_nrmse, cv = 10))
        model_name = 'ridge_'+model_name
    elif args.regressor == 'hgb':
        model = SimpleStackingRegressor(estimators, HistGradientBoostingRegressorCV(scoring = neg_nrmse, cv = 10, verbose = 3))
        model_name = 'hgb_'+model_name
    else:
        raise NotImplementedError(f'Regressor {args.regressor} not implemented yet.')

    print('Fitting model')
    model.fit(X, y)
    
    print('Testing model')
    if hasattr(model.final_estimator, 'best_score_'):
        print(f'NRMSE test (CV10): {-model.final_estimator.best_score_:.3e}')
    else:
        _y = model.predict(X)
        print(f'NRMSE test: {nrmse(y, _y):.3e}')
    
    print('Saving model')
    TO = config.results_root() / 'model'
    TO.mkdir(parents=True, exist_ok=True)
    dump(model, TO/f"{model_name}.pkl")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test sklearn model on challenge 1')
    parser.add_argument("model", nargs = '+', type=Path, help="Path to model file (.pkl)")
    parser.add_argument("--name", type=str, default = None, help="Path to model file (.pkl)")
    parser.add_argument("--regressor", type=str, default = 'mean', help="Regressor")
    parser.add_argument("-k", type=int, default = None, help="How many in the list")
    args = parser.parse_args()
    main(args)