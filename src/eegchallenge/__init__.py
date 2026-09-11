"""Stanford SCSNL entry to the NeurIPS 2025 EEG Foundation Challenge.

Two tracks share this library:

* a deep-learning track (braindecode backbones + PyTorch Lightning) reached
  through :mod:`eegchallenge.models`, and
* a scikit-learn track (engineered spectral features + gradient boosting)
  reached through :mod:`eegchallenge.features` and :mod:`eegchallenge.pipeline`.

Filesystem locations come from environment variables — see
:mod:`eegchallenge.config` and ``config/paths.example.sh``.
"""

__version__ = "1.0.0"
