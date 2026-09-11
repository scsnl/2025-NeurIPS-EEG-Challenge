"""Feature transforms for the scikit-learn track.

These are the candidate representations searched over in ``experiments/``. Each
takes a trial array and returns a transformed array; they are used inside
``sklearn.pipeline.Pipeline`` via ``FunctionTransformer``, so they are plain
functions with no fitted state.

Unless noted, input is ``(trials, channels, time) == (n, 129, 200)`` — 2 s
windows at 100 Hz across the 129-channel HBN montage. The shape assertions are
deliberate: a silent shape change here would quietly corrupt a grid search.

Extracted verbatim from ``challenges/challenge2/supervised_linear.py``, which
despite its location was shared by both challenges.
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

FS = 100
NPERSEG = 100

__all__ = [
    "polynomial",
    "enveloppe",
    "complex_cepstrum",
    "real_cepstrum",
    "power_phase_stft",
    "log10_safe",
    "psd",
    "power_phase_circular",
    "power_phase",
    "power_phase_temporal",
    "add_derivative",
    "functional_connectivity",
    "bandpower_canonical",
    "mad_scaling_clipping",
    "mad_scaling_clipping_channel",
    "mad_scaling_clipping_channel_time",
    "mad_scaling_clipping_channel_trial",
    "mad_scaling_clipping_trial_time",
    "volts_to_microvolts",
    "reshape",
]


def polynomial(X):
    assert len(X.shape)==2
    X_polynomial = np.concatenate((X, X**2), axis=-1)
    return X_polynomial


def enveloppe(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_f = signal.hilbert(X, axis = -1)
    X_enveloppe = np.abs(X_f)
    X_phase = np.angle(X_f)
    X_enveloppe_phase = np.concatenate((X_enveloppe,X_phase),axis=1)
    assert len(X_enveloppe_phase.shape)==3
    assert X_enveloppe_phase.shape[1:3]==(258,200)
    return X_enveloppe_phase


def complex_cepstrum(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_f = rfft(X, axis = -1)
    X_power = np.abs(X_f)
    X_power = np.clip(X_power, 1e-12, None)
    X_phase = np.angle(X_f)
    X_c = np.fft.irfft( np.log(X_power) + 1j * np.unwrap(X_phase, axis = -1), n=X.shape[-1], axis = -1)
    assert X_c.shape[1:]==(129, 200)
    return X_c


def real_cepstrum(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_f = rfft(X, axis = -1)
    X_power = np.abs(X_f)
    X_power = np.clip(X_power, 1e-12, None)
    X_c = np.fft.irfft( np.log(X_power), n=X.shape[-1], axis = -1)
    assert X_c.shape[1:]==(129, 200)
    return X_c


def power_phase_stft(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    freqs, time, X_f = signal.stft(X, fs = 100, nperseg = 20, axis = -1)
    X_power = np.abs(X_f)
    X_phase = np.angle(X_f)
    assert X_power.shape[1:]==X_phase.shape[1:]==(129, 11, 21)
    X_power_phase = np.concatenate((X_power, X_phase), axis=-1)
    assert X_power_phase.shape[1:]==(129, 11, 21*2)
    return X_power_phase


def log10_safe(X):
    X = np.asarray(X, dtype=np.float64)
    eps = np.finfo(X.dtype).tiny
    return np.log10(np.clip(X, eps, None))


def psd(X):
    assert len(X.shape)==3
    assert X.shape[1]==129
    assert X.shape[2]==200
    fs = []
    Pxxs = []
    for channel_idx in range(X.shape[1]):
        f, Pxx = signal.welch(X[:,channel_idx,:], fs=FS, axis=1, nperseg=NPERSEG)
        fs.append(f)
        Pxxs.append(Pxx)
    Pxxs = np.array(Pxxs)
    Pxxs = np.swapaxes(Pxxs,0,1)
    return Pxxs


def power_phase_circular(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_f = rfft(X, axis = -1)
    X_power = np.abs(X_f)
    X_phase = np.angle(X_f)
    X_power_phase = np.concatenate((X_power, np.sin(X_phase), np.cos(X_phase)), axis=-1)
    assert len(X_power_phase.shape)==3
    assert X_power_phase.shape[1:3]==(129,303)
    return X_power_phase


def power_phase(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_f = rfft(X, axis = -1)
    X_power = np.abs(X_f)
    X_phase = np.angle(X_f)
    X_power_phase = np.concatenate((X_power,X_phase),axis=-1)
    assert len(X_power_phase.shape)==3
    assert X_power_phase.shape[1:3]==(129,202)
    return X_power_phase


def power_phase_temporal(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_f = rfft(X, axis = -1)
    X_power = np.abs(X_f)
    X_phase = np.angle(X_f)
    X_power_phase = np.concatenate((X_power,X_phase),axis=-1)
    assert len(X_power_phase.shape)==3
    assert X_power_phase.shape[1:3]==(129,202)
    X_power_phase_temporal = np.concatenate((X_power_phase, X), axis=-1)
    assert len(X_power_phase_temporal.shape)==3
    assert X_power_phase_temporal.shape[1:3]==(129,402)
    return X_power_phase_temporal


def add_derivative(X, n=1):
    assert len(X.shape)==3
    assert X.shape[1]==129
    derivative = np.diff(X, axis=-1, n=n)
    assert X.shape[0:2]==derivative.shape[0:2]
    assert (X.shape[2]-n)==derivative.shape[2]
    X_derivative = np.concatenate((X, derivative), axis=-1)
    assert len(X_derivative.shape)==3
    assert X.shape[0:2]==derivative.shape[0:2]==X_derivative.shape[0:2]
    return X_derivative


def functional_connectivity(X):
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    fcs = []
    for i in range(X.shape[0]):
        fc = np.corrcoef(X[i])
        fc = fc[np.tril_indices(fc.shape[0],-1)]
        fcs.append(fc)
    fcs = np.array(fcs)
    fcs = np.nan_to_num(fcs)
    assert len(fcs.shape)==2
    assert fcs.shape[0]==X.shape[0]
    assert fcs.shape[1]==(X.shape[1]*(X.shape[1]-1)/2)
    return fcs


def bandpower_canonical(X, fs=FS, bands=None):
    """
    X: (trials, channels, time)
    returns: (trials, channels, n_bands) band power (integral of one-sided PSD)
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (trials, channels, time), got {X.shape}")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    if bands is None:
        bands = [("delta",(0.5,4)), ("theta",(4,7)), ("alpha",(8,13)), ("beta",(13,30)), ("gamma",(30,80))]

    N = X.shape[-1]

    f   = rfftfreq(N, d=1.0/fs)              # (F,)
    Xf  = rfft(X, axis=-1)                   # (Ntr, Nch, F)
    Pxx = (Xf.real**2 + Xf.imag**2) / (fs*N) # raw one-sided periodogram
    # Double non-DC (and non-Nyquist) bins to conserve power
    if N % 2 == 0:                           # Nyquist present
        if Pxx.shape[-1] > 2: Pxx[..., 1:-1] *= 2.0
    else:
        if Pxx.shape[-1] > 1: Pxx[..., 1:]  *= 2.0

    df  = fs / N
    nyq = f[-1]

    out = []
    for _, (lo, hi) in bands:
        lo = max(0.0, lo)        # clamp to valid range
        hi = min(nyq, hi)
        if hi <= lo:
            out.append(np.zeros(X.shape[:2], dtype=Pxx.dtype))
            continue
        # include Nyquist only in a band that reaches it
        mask = (f >= lo) & ((f < hi) if hi < nyq else (f <= nyq))
        out.append(Pxx[..., mask].sum(-1) * df if mask.any()
                   else np.zeros(X.shape[:2], dtype=Pxx.dtype))

    return np.stack(out, axis=-1)


def mad_scaling_clipping_channel(X):
    """Mean absolute deviation scaling and clipping per channel (over trials and time)."""
    return mad_scaling_clipping(X, (0,2))


def mad_scaling_clipping_channel_time(X):
    """Mean absolute deviation scaling and clipping per channel-time (over trials)."""
    return mad_scaling_clipping(X, 0)


def mad_scaling_clipping_channel_trial(X):
    """Mean absolute deviation scaling and clipping per channel-trial (over time)."""
    return mad_scaling_clipping(X, 2)


def mad_scaling_clipping_trial_time(X):
    """Mean absolute deviation scaling and clipping per trial-time (over channels)."""
    return mad_scaling_clipping(X, 1)


def mad_scaling_clipping(X, axis):
    """Mean absolute deviation scaling and clipping."""
    assert len(X.shape)==3
    assert X.shape[1:3]==(129,200)
    X_med = np.median(X, axis=axis, keepdims=True)
    X_mad = np.median(np.abs(X-X_med), axis=axis, keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'):
        X_scaled_clipped = np.clip((X-X_med)/(X_mad/0.6745), -5, 5)
    X_scaled_clipped = np.nan_to_num(X_scaled_clipped)
    return X_scaled_clipped


def volts_to_microvolts(X):
    return 1e6 * X


def reshape(X):
    return X.reshape(X.shape[0],-1)
