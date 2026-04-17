from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
DEFAULT_MODEL_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"


def get_model_path(model_path: str | None = None) -> str:
    if model_path is None:
        model_path = Path(__file__).parent / "rmvpe.pt"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"RMVPE model not found at {model_path}")
        print(f"Downloading from {DEFAULT_MODEL_URL}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(model_path))
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")

    return str(model_path)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def _get_hann_window(self, keyshift, win_length_new):
        if keyshift not in self.hann_window:
            self.hann_window[keyshift] = torch.hann_window(win_length_new)
        return self.hann_window[keyshift]

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(self.n_fft * factor)
        win_length_new = int(self.win_length * factor)
        hop_length_new = int(self.hop_length * speed)

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self._get_hann_window(keyshift, win_length_new),
            center=center,
            return_complex=True,
        )
        magnitude = torch.abs(fft)

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if self.is_shortcut else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        if stride == (1, 2):
            out_padding = (0, 1)
        elif stride == (2, 2):
            out_padding = (1, 1)
        elif stride == (2, 1):
            out_padding = (1, 0)
        else:
            out_padding = (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            skip, x = self.layers[i](x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)])
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        return self.gru(x)[0]


class RMVE(nn.Module):
    def __init__(self, hop_length, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Sigmoid())

    def forward(self, x):
        mel = self.mel(x.reshape(-1, x.shape[-1]))
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.unet(mel)
        if n_pad > 0:
            x = x[:, :, :-n_pad, :]
        x = self.cnn(x).transpose(1, 2).flatten(-2)
        return self.fc(x)


def to_local_average_cents(salience, center=None, thred=0.0):
    if not hasattr(to_local_average_cents, "cents_mapping"):
        to_local_average_cents.cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], None, thred) for i in range(salience.shape[0])])
    raise ValueError("label should be either 1d or 2d ndarray")


class RMVEPitchAlgorithm:

    def __init__(self, model_path: str | None = None, sample_rate: int = SAMPLE_RATE, hop_size: int = 160, fmin: float = MEL_FMIN, fmax: float = MEL_FMAX):
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

        self.model_hop_length = 160
        model_path = get_model_path(model_path)
        self.load_model(model_path)

    def load_model(self, model_path):
        model = RMVE(self.model_hop_length, 4, 1, (2, 2))
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if hasattr(checkpoint, "module"):
            state_dict = checkpoint.module.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        model.eval()
        self.model = model

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if self.sample_rate != SAMPLE_RATE:
            try:
                from resampy import resample

                audio = resample(audio, self.sample_rate, SAMPLE_RATE)
            except ImportError:
                from scipy.signal import resample

                target_length = int(len(audio) * SAMPLE_RATE / self.sample_rate)
                audio = resample(audio, target_length).astype(np.float32)

        return audio

    def _extract_raw_pitch_and_periodicity(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio_processed = self._preprocess_audio(audio)
        audio_tensor = torch.from_numpy(audio_processed).float().contiguous()

        with torch.no_grad():
            pitch_pred = self.model(audio_tensor.unsqueeze(0)).squeeze(0)

        pitch_pred_np = pitch_pred.cpu().numpy()
        del audio_tensor
        del pitch_pred

        cents = to_local_average_cents(pitch_pred_np, thred=0.0)
        f0 = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents])
        periodicity = np.max(pitch_pred_np, axis=1) if pitch_pred_np.ndim > 1 else pitch_pred_np
        model_hopsize_seconds = self.model_hop_length / SAMPLE_RATE
        n_frames = len(f0)
        times = np.arange(n_frames) * model_hopsize_seconds
        return times, f0, periodicity


    def _get_default_threshold(self) -> float:
        return 0.03

    def extract_continuous_periodicity(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times, pitch, periodicity = self._extract_raw_pitch_and_periodicity(audio)
        pitch, periodicity = self._sanity_check(pitch, periodicity)
        target_times = self._compute_target_times(len(audio))
        aligned_pitch = self._align_to_grid(times, pitch, target_times)
        aligned_periodicity = self._align_to_grid(times, periodicity, target_times)
        return aligned_pitch, aligned_periodicity

    def _compute_target_times(self, audio_length: int) -> np.ndarray:
        n_hops = audio_length // self.hop_size
        return np.arange(n_hops) * (self.hop_size / self.sample_rate)

    def _align_to_grid(
        self,
        algorithm_times: np.ndarray,
        values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        if len(algorithm_times) == 0:
            return np.zeros_like(target_times)
        return np.interp(target_times, algorithm_times, values, left=0.0, right=0.0)

    def _sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)

        voiced = periodicity > 0
        pitch[~voiced] = 0.0
        pitch[voiced] = np.clip(pitch[voiced], self.fmin, self.fmax)

        periodicity = np.clip(periodicity, 0.0, 1.0)
        return pitch, periodicity
