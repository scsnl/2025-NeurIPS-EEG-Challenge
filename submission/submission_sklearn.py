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
# import psutil
import time
import datetime as _dt
import os, sys

def get_cpu_count():
    return os.cpu_count() or 1

def get_total_ram_gb():
    # Prefer POSIX sysconf on Linux
    if sys.platform.startswith("linux") \
       and ("SC_PAGE_SIZE" in os.sysconf_names) \
       and ("SC_PHYS_PAGES" in os.sysconf_names):
        page_size = os.sysconf("SC_PAGE_SIZE")      # bytes
        phys_pages = os.sysconf("SC_PHYS_PAGES")    # count
        if isinstance(page_size, int) and isinstance(phys_pages, int) \
           and page_size > 0 and phys_pages > 0:
            return (page_size * phys_pages) / (1024**3)

    # Fallback: /proc/meminfo (Linux)
    p = Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                # parts[1] is kB per /proc/meminfo format
                if len(parts) >= 2 and parts[1].isdigit():
                    kb = int(parts[1])
                    return kb / (1024**2)

    # If everything else fails, return NaN-like
    return float("nan")

def _utc_now_iso():
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

class SklearnToTorch():
    def __init__(self, model):
        self.model = model
        self.start_time = None  # set by Submission

    def forward(self, x):
        y = torch.from_numpy(self.model.predict(x.detach().cpu().numpy())[:, None])

        # Print time right before returning
        now = _utc_now_iso()
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"[{now}] elapsed_since_start={elapsed:.3f}s", flush=True)
        else:
            print(f"[{now}] forward()", flush=True)
        return y

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

# put near the top of submission.py, before joblib.load(...)
def _patch_pathlib_for_unpickle():
    import sys, importlib
    # If a non-package named 'pathlib' is shadowing stdlib, try to remove it
    if 'pathlib' in sys.modules and not hasattr(sys.modules['pathlib'], 'Path'):
        sys.modules.pop('pathlib', None)
    # Import stdlib pathlib
    std_pathlib = importlib.import_module('pathlib')
    # Provide an alias for the missing submodule that the pickle expects
    sys.modules['pathlib._local'] = std_pathlib

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.start_time = time.time()  # start clock here
        self.sfreq = SFREQ
        self.device = DEVICE
        add_python_packages()
        # cpu_count = psutil.cpu_count(logical=True)
        # ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = get_cpu_count()
        ram_gb = get_total_ram_gb()
        print(cpu_count)
        print(ram_gb)

    def get_model_challenge_1(self):
        _patch_pathlib_for_unpickle()
        f = resolve_path("challenge_1.pkl")
        model_challenge1 = SklearnToTorch(joblib.load(f))
        if hasattr(model_challenge1.model, "memory"):
            model_challenge1.model.memory = None
        model_challenge1.start_time = self.start_time  # pass down
        return model_challenge1

    def get_model_challenge_2(self):
        _patch_pathlib_for_unpickle()
        f = resolve_path("challenge_2.pkl")
        model_challenge2 = SklearnToTorch(joblib.load(f))
        if hasattr(model_challenge2.model, "memory"):
            model_challenge2.model.memory = None
        model_challenge2.start_time = self.start_time  # pass down
        return model_challenge2

# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()
#
# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         y_pred = model_1.forward(X)
#         ...
# del model_1
# gc.collect()
#
# model_2 = sub.get_model_challenge_2()
# model_2.eval()
# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         y_pred = model_2.forward(X)
#         ...
# del model_2
# gc.collect()
#
# score1 = compute_score_challenge_1(y_true, y_preds)
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)
