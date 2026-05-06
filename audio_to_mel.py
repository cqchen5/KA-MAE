import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    """Generate a Mel filterbank matrix (equivalent to librosa.filters.mel)."""
    if fmax is None:
        fmax = sr / 2

    # Mel scale conversion functions
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # Compute mel scale points
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin frequencies
    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Initialize filterbank
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = fft_bins[m - 1]   # left
        f_m = fft_bins[m]             # center
        f_m_plus = fft_bins[m + 1]    # right

        if f_m_minus != f_m:
            filterbank[m - 1, f_m_minus:f_m] = (
                np.arange(f_m_minus, f_m) - f_m_minus
            ) / (f_m - f_m_minus)
        if f_m != f_m_plus:
            filterbank[m - 1, f_m:f_m_plus] = (
                f_m_plus - np.arange(f_m, f_m_plus)
            ) / (f_m_plus - f_m)

    return filterbank

class Audio2Mel(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024,
                 sampling_rate=22050, n_mel_channels=80, mel_fmin=0.0, mel_fmax=None,
                 device='cuda'):
        super().__init__()
        window = torch.hann_window(win_length, device=device).float()
        # 假设你已有 mel_basis 生成函数，或者用之前不依赖 librosa 的版本
        mel_basis = torch.from_numpy(mel_filterbank(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax
        )).to(device=device, dtype=torch.float32)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audioin):
        shape = audioin.shape
        if len(shape) > 2:
            audioin = audioin.reshape(shape[0] * shape[1], -1)
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect")

        # 使用复数输出（更直观）
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,   # 注意：需要 PyTorch 1.8+ 支持
        )  # shape: (batch, n_fft//2+1, complex)

        power_spec = fft.abs() ** 2  # magnitude squared, 非负
        # mel_basis: (n_mels, n_fft//2+1), power_spec: (B, n_fft//2+1, T)
        # 需把 power_spec 的 time dim 放到最后再矩阵乘
        # 如果 power_spec 形状是 (B, F, T)，我们想得到 (B, n_mels, T)
        mel_output = torch.matmul(self.mel_basis, power_spec)  # (n_mels, ) x (B,F,T) -> (B, n_mels, T)

        # 把可能的 NaN/Inf 清理掉
        mel_output = torch.nan_to_num(mel_output, nan=1e-4, posinf=1e4, neginf=1e-4)
        # 最后 clamp 避免 log(0)
        mel_output = torch.clamp(mel_output, min=1e-4, max=1e4)

        log_mel_spec = torch.log10(mel_output)

        if len(shape) > 2:
            log_mel_spec = log_mel_spec.reshape(shape[0], shape[1], -1)
        return log_mel_spec
