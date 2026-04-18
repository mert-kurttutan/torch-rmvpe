from __future__ import annotations

from pathlib import Path

import torch
from torch_rmve import RMVEPitchAlgorithm


def test_random_audio_shapes() -> None:
    algorithm = RMVEPitchAlgorithm(sample_rate=16000, hop_size=160)
    batch_size = 1
    sample_rate = 16000
    seconds = 30
    audio = torch.rand(batch_size, sample_rate * seconds) * 2.0 - 1.0

    pitch, periodicity = algorithm.extract_continuous_periodicity(audio)

    # expected_frames = len(audio) // algorithm.hop_size
    expected_frames = (audio.shape[1] + algorithm.hop_size - 1) // algorithm.hop_size
    assert pitch.shape == (batch_size, expected_frames)
    assert periodicity.shape == (batch_size, expected_frames)
