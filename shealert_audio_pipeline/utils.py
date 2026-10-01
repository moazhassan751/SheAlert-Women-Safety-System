"""
utils.py
--------
Shared helpers used by every dataset loader: duration probing, SNR
estimation, and the QC gate from Section 10 of the scope doc
(reject <3s clips, reject clipped clips, reject clips below the SNR floor).
"""

from pathlib import Path
import re
import numpy as np
import soundfile as sf
import librosa


def to_relative_path(path: str | Path, base_path: str | Path = None) -> str:
    """
    Convert an absolute or relative path to a path relative to base_path,
    normalized with forward slashes (e.g. 'UrduSpeech/Distress/clip.wav').
    """
    if base_path is None:
        import config
        base_path = config.BASE_PATH
    p = Path(path)
    b = Path(base_path).resolve()
    try:
        rel = p.resolve().relative_to(b)
        return rel.as_posix()
    except (ValueError, RuntimeError):
        return str(path).replace("\\", "/")


def resolve_relative_path(rel_path: str | Path, base_path: str | Path = None) -> Path:
    """
    Turn a relative path into an absolute Path using base_path.
    If the path is already absolute, returns it resolved.
    """
    if base_path is None:
        import config
        base_path = config.BASE_PATH
    p = Path(rel_path)
    if p.is_absolute():
        return p.resolve()
    return (Path(base_path) / p).resolve()


resolve_path = resolve_relative_path


def get_rejection_category(reason: str) -> str:
    """
    Standardize QC rejection reason string into a clean category name:
    low_snr, clipped, too_short, unreadable.
    """
    if not reason:
        return ""
    r = reason.lower()
    if "too short" in r or "too_short" in r:
        return "too_short"
    elif "clipped" in r:
        return "clipped"
    elif "low snr" in r or "low_snr" in r:
        return "low_snr"
    elif "unreadable" in r:
        return "unreadable"
    return reason.split()[0].split(":")[0]


def extract_group_id(filepath: str | Path, source_dataset: str) -> tuple[str, bool]:
    """
    Extracts group_id for speaker-grouped splitting.
    Returns (group_id, is_per_file_fallback).
    All group_ids are prefixed with source_dataset to guarantee no inter-dataset collisions.
    """
    p = Path(filepath)
    stem = p.stem
    name = p.name
    src = str(source_dataset).lower()

    if src == "urduspeech":
        # e.g. SPEAKER_0169_DRAMA_001273.wav -> urduspeech_SPEAKER_0169
        m = re.search(r"SPEAKER_\d+", name, re.IGNORECASE)
        if m:
            return f"urduspeech_{m.group(0).upper()}", False
        return f"urduspeech_file_{stem}", True

    elif src == "semour":
        # SEMOUR+ structure: root/Actor_N/Emotion/*.wav
        actor_match = None
        for part in p.parts:
            if re.match(r"^actor_\d+$", part, re.IGNORECASE):
                actor_match = part.lower()
                break
        if actor_match:
            return f"semour_{actor_match}", False
        return f"semour_file_{stem}", True

    elif src == "cremad":
        # e.g. 1001_DFA_ANG_XX.mp3 -> ActorID is parts[0]
        parts = stem.split("_")
        if len(parts) >= 1 and parts[0].isdigit():
            return f"cremad_{parts[0]}", False
        return f"cremad_file_{stem}", True

    elif src == "ravdess":
        # 03-01-06-01-02-01-12.wav -> 7th field is Actor (e.g. 12)
        fields = stem.split("-")
        if len(fields) == 7 and fields[-1].isdigit():
            return f"ravdess_actor_{int(fields[-1]):02d}", False
        for part in p.parts:
            if re.match(r"^actor_\d+$", part, re.IGNORECASE):
                return f"ravdess_{part.lower()}", False
        return f"ravdess_file_{stem}", True

    elif src == "urduser":
        # 10_1_1_01.wav -> first number is Actor ID (1 to 10)
        parts = stem.split("_")
        if len(parts) >= 1 and parts[0].isdigit():
            return f"urduser_actor_{int(parts[0])}", False
        return f"urduser_file_{stem}", True

    elif src == "urdu_dataset_master":
        # SM1_F4_A04.wav -> first token is speaker code (e.g. SM1, SF4)
        parts = stem.split("_")
        if len(parts) >= 1 and re.match(r"^S[MF]\d+$", parts[0], re.IGNORECASE):
            return f"urdu_dataset_master_{parts[0].upper()}", False
        return f"urdu_dataset_master_file_{stem}", True

    elif src == "audioset_scream":
        # -20uudT97E0_30.00_40.00.wav -> YouTube video ID is first part before _start_end
        parts = stem.rsplit("_", 2)
        if len(parts) == 3:
            return f"audioset_scream_{parts[0]}", False
        return f"audioset_scream_file_{stem}", True

    elif src == "esc50":
        # 1-100032-A-0.wav -> fold-ID-take-class -> ID is Freesound ID (field 1)
        parts = stem.split("-")
        if len(parts) >= 2:
            return f"esc50_{parts[1]}", False
        return f"esc50_file_{stem}", True

    elif src == "urbansound8k":
        # 101415-3-0-2.wav -> Freesound ID is field 0
        parts = stem.split("-")
        if len(parts) >= 1:
            return f"urbansound8k_{parts[0]}", False
        return f"urbansound8k_file_{stem}", True

    elif src == "realworld_urdu":
        # WhatsApp audio files, no speaker metadata -> per-file fallback
        return f"realworld_urdu_{stem}", True

    else:
        return f"{src}_file_{stem}", True



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


def is_clipped(
    y: np.ndarray,
    threshold: float = None,
    max_fraction: float = None,
) -> bool:
    """True if more than max_fraction of samples sit at/above the clipping threshold."""
    if threshold is None:
        import config
        threshold = getattr(config, "CLIPPING_THRESHOLD", 0.999)
    if max_fraction is None:
        import config
        max_fraction = getattr(config, "CLIPPING_MAX_FRACTION", 0.001)

    clipped_samples = np.sum(np.abs(y) >= threshold)
    return (clipped_samples / len(y)) > max_fraction if len(y) else True


def passes_qc(
    filepath,
    min_duration: float,
    min_snr_db: float,
    return_duration: bool = False,
    source_dataset: str = None,
    return_clip_frac: bool = False,
):
    """
    Runs the full QC gate on one file. Returns:
      (passed, reason_if_rejected) if return_duration=False and return_clip_frac=False
      (passed, reason_if_rejected, duration) if return_duration=True and return_clip_frac=False
      (passed, reason_if_rejected, duration, clip_frac) if return_clip_frac=True
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
            res = (False, f"unreadable: {e}")
            if return_clip_frac:
                return (*res, 0.0, 0.0)
            return (*res, 0.0) if return_duration else res

    duration = len(y) / sr

    # Compute clip_frac (fraction of samples >= CLIPPING_THRESHOLD)
    import config
    clip_thresh = getattr(config, "CLIPPING_THRESHOLD", 0.999)
    clip_max_frac = getattr(config, "CLIPPING_MAX_FRACTION", 0.001)
    exempt_datasets = getattr(config, "CLIP_EXEMPT_DATASETS", [])

    clipped_samples = int(np.sum(np.abs(y) >= clip_thresh))
    clip_frac = float(clipped_samples / len(y)) if len(y) else 0.0

    if duration < min_duration:
        res = (False, f"too short ({duration:.2f}s < {min_duration}s)")
        if return_clip_frac:
            return (*res, duration, clip_frac)
        return (*res, duration) if return_duration else res

    is_exempt = source_dataset in exempt_datasets
    if not is_exempt and clip_frac > clip_max_frac:
        res = (False, "clipped")
        if return_clip_frac:
            return (*res, duration, clip_frac)
        return (*res, duration) if return_duration else res

    snr = estimate_snr_db(y)
    if snr < min_snr_db:
        res = (False, f"low SNR ({snr:.1f}dB < {min_snr_db}dB)")
        if return_clip_frac:
            return (*res, duration, clip_frac)
        return (*res, duration) if return_duration else res

    res = (True, "")
    if return_clip_frac:
        return (*res, duration, clip_frac)
    return (*res, duration) if return_duration else res


