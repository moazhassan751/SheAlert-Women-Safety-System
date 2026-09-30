"""
utils.py
--------
Shared helpers used by every dataset loader: duration probing, SNR
estimation, and the QC gate from Section 10 of the scope doc
(reject <3s clips, reject clipped clips, reject clips below the SNR floor).
"""

import numpy as np
import soundfile as sf
import librosa


def get_duration_sec(filepath) -> float:
    """Return duration in seconds. Tries soundfile first (fast, no decode);
    falls back to librosa for formats soundfile can't parse (e.g. some mp3
    builds) — CREMA-D ships as .mp3, so this fallback matters."""
    try:
        info = sf.info(str(filepath))
        return info.frames / info.samplerate
    except Exception:
        return librosa.get_duration(path=str(filepath))


def estimate_snr_db(y: np.ndarray, frame_len: int = 2048, hop: int = 512) -> float:
    """
    Rough frame-energy-based SNR estimate: treats the 10th percentile of
    frame energy as the noise floor and the 95th percentile as signal peak.
    Good enough for a QC gate; not a substitute for a proper VAD-based SNR.
    """
    if len(y) < frame_len:
        return 0.0
    energies = np.array([
        np.sum(y[i:i + frame_len] ** 2)
        for i in range(0, len(y) - frame_len, hop)
    ])
    energies = energies[energies > 0]
    if len(energies) == 0:
        return 0.0
    noise_floor = np.percentile(energies, 10)
    signal_peak = np.percentile(energies, 95)
    if noise_floor == 0:
        noise_floor = 1e-10
    return float(10 * np.log10(signal_peak / noise_floor))


def is_clipped(y: np.ndarray, threshold: float = 0.999, max_fraction: float = 0.001) -> bool:
    """True if more than max_fraction of samples sit at/above the clipping threshold."""
    clipped_samples = np.sum(np.abs(y) >= threshold)
    return (clipped_samples / len(y)) > max_fraction if len(y) else True


def passes_qc(filepath, min_duration: float, min_snr_db: float) -> tuple[bool, str]:
    """
    Runs the full QC gate on one file. Returns (passed, reason_if_rejected).
    Loads the waveform once (needed for clipping + SNR checks).
    """
    try:
        y, sr = sf.read(str(filepath), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)  # downmix to mono for the check
    except Exception:
        try:
            y, sr = librosa.load(str(filepath), sr=None, mono=True)
        except Exception as e:
            return False, f"unreadable: {e}"

    duration = len(y) / sr
    if duration < min_duration:
        return False, f"too short ({duration:.2f}s < {min_duration}s)"

    if is_clipped(y):
        return False, "clipped"

    snr = estimate_snr_db(y)
    if snr < min_snr_db:
        return False, f"low SNR ({snr:.1f}dB < {min_snr_db}dB)"

    return True, ""
