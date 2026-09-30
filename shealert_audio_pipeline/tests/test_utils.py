import numpy as np
import pytest
import soundfile as sf
from utils import get_duration_sec, estimate_snr_db, is_clipped, passes_qc

def test_estimate_snr_burst():
    sr = 16000
    # 0.5s silence + 0.5s tone + 0.5s silence (typical speech-like utterance structure)
    silence = np.random.normal(0, 0.001, int(0.5 * sr))
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.001, int(0.5 * sr))
    signal = np.concatenate([silence, tone, silence])
    
    snr = estimate_snr_db(signal)
    assert snr > 10.0, f"Expected high SNR for speech-like burst, got {snr}"

def test_is_clipped():
    normal_signal = np.sin(np.linspace(0, 10, 1000)) * 0.5
    assert not is_clipped(normal_signal)

    clipped_signal = np.ones(1000)
    assert is_clipped(clipped_signal)

def test_qc_with_synthetic_audio(tmp_path):
    sr = 16000
    # Signal with silence and tone burst
    silence = np.random.normal(0, 0.001, int(0.5 * sr)).astype(np.float32)
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = np.concatenate([silence, tone, silence])

    good_wav = tmp_path / "good.wav"
    sf.write(str(good_wav), y, sr)

    # 1. Good file passes QC
    ok, reason = passes_qc(good_wav, min_duration=0.5, min_snr_db=3.0)
    assert ok, f"Good audio rejected: {reason}"

    # 2. Too short file fails duration check
    ok_short, reason_short = passes_qc(good_wav, min_duration=3.0, min_snr_db=3.0)
    assert not ok_short
    assert "too short" in reason_short

    # 3. Non-existent file fails unreadable
    ok_missing, reason_missing = passes_qc(tmp_path / "missing.wav", min_duration=0.5, min_snr_db=3.0)
    assert not ok_missing
    assert "unreadable" in reason_missing
