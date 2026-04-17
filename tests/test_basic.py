from __future__ import annotations

from pathlib import Path

import numpy as np

from torch_rmve import RMVEPitchAlgorithm


def test_random_audio_shapes() -> None:
    algorithm = RMVEPitchAlgorithm(sample_rate=16000, hop_size=160)
    audio = np.random.rand(16000).astype(np.float32) * 2.0 - 1.0

    pitch, periodicity = algorithm.extract_continuous_periodicity(audio)

    print("audio shape:", audio.shape)
    print("pitch shape:", pitch.shape)
    print("periodicity shape:", periodicity.shape)

    expected_frames = len(audio) // algorithm.hop_size
    assert pitch.shape == (expected_frames,)
    assert periodicity.shape == (expected_frames,)
