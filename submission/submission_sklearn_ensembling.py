# ##########################################################################
# # Example of submission files
# # ---------------------------
# The zip file needs to be single level depth!
# NO FOLDER
# my_submission.zip
# ├── submission.py
# ├── challenge_1.pkl
# ├── challenge_2.pkl
# └── python_packages
#     ├── challenges
#     │   └── challenge2
#     │       └── supervised_linear.py
#     └── utils
#         └── submit.py

import torch
from pathlib import Path
import joblib
import sys, importlib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
import psutil

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

class SklearnToTorch():
    def __init__(self, model):
        self.model = model
    
    def forward(self, x):
        return torch.from_numpy(self.model.predict(x.detach().cpu().numpy())[:,None])

    def eval(self):
        return

def resolve_path(name="model_file_name"):
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory"
        )

def add_python_packages():
    # Only append if present; otherwise do nothing
    try:
        sys.path.append(resolve_path("python_packages"))
    except FileNotFoundError:
        pass

def _patch_pathlib_for_unpickle():
    # If a non-package named 'pathlib' is shadowing stdlib, try to remove it
    if 'pathlib' in sys.modules and not hasattr(sys.modules['pathlib'], 'Path'):
        sys.modules.pop('pathlib', None)
    # Import stdlib pathlib
    std_pathlib = importlib.import_module('pathlib')
    # Provide an alias for the missing submodule that the pickle expects
    sys.modules['pathlib._local'] = std_pathlib

def _register_classes_for_unpickle():
    """
    Make classes available under __main__ so pickles saved with module='__main__'
    can be loaded inside the evaluator where __main__ is ingest_score.py.
    """
    main_mod = sys.modules.get('__main__')
    if main_mod is None:
        return
    for cls in (SimpleStackingRegressor, MeanRegressor):
        setattr(main_mod, cls.__name__, cls)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        add_python_packages()
        cpu_count = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(cpu_count)
        print(ram_gb)

    def get_model_challenge_1(self):
        _patch_pathlib_for_unpickle()
        _register_classes_for_unpickle()
        f = resolve_path("challenge_1.pkl")
        model_challenge1 = SklearnToTorch(joblib.load(f))
        model_challenge1.model.memory=None
        return model_challenge1

    def get_model_challenge_2(self):
        _patch_pathlib_for_unpickle()
        _register_classes_for_unpickle()
        f = resolve_path("challenge_2.pkl")
        model_challenge2 = SklearnToTorch(joblib.load(f))
        model_challenge2.model.memory=None
        return model_challenge2

# if __name__ == '__main__':
#     model = Submission(100, 'cpu').get_model_challenge_1()
#     print(model.model)
#     print(model.forward(torch.zeros((1, 129, 200))).shape)
#     breakpoint()


# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()

# warmup_loader_challenge_1 = DataLoader(HBN_R5_dataset1, batch_size=BATCH_SIZE)
# final_loader_challenge_1 = DataLoader(secret_dataset1, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:  # and final_loader later
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_1.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score1 = compute_score_challenge_1(y_true, y_preds)
# del model_1
# gc.collect()

# model_2 = sub.get_model_challenge_2()
# model_2.eval()

# warmup_loader_challenge_2 = DataLoader(HBN_R5_dataset2, batch_size=BATCH_SIZE)
# final_loader_challenge_2 = DataLoader(secret_dataset2, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:  # and final_loader later
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_2.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)
