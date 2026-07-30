"""Exact integer-phase harmonic source (opt-in alternative to SineGen).

SineGen accumulates phase with an unbounded float32 cumsum and synthesizes each
harmonic with an independent accumulator. Two consequences:

* The accumulated phase grows without bound (~113k rad after 10 s at F0=200 Hz for
  the 9th harmonic). In float16 the ULP at that magnitude exceeds the per-sample
  phase increment, which audibly scrambles the sines — half-precision deployments
  currently must keep the decoder in float32.
* The per-harmonic accumulators round independently, so harmonics slowly drift
  away from exact integer ratios of the fundamental.

ExactSineGen replaces the accumulator with integer modular arithmetic:

    inc    = round(f0 / fs * Q),  Q = 2**48
    phi[n] = (phi[n-1] + inc) mod Q          # int64, blockwise carry, any duration
    phi_k  = (k * phi) mod Q                 # harmonic k phase-locked to the fundamental
    sine   = sin(2*pi * phi / Q)             # bounded argument, half-precision safe

Properties versus SineGen:

* zero accumulated phase error for any utterance length;
* harmonics exactly at integer multiples of the fundamental by construction;
* no down/up x``upsample_scale`` interpolation round trip, hence no built-in
  half-window excitation delay relative to the frame-aligned conditioning;
* deterministic (no random initial phases);
* the sine argument stays bounded, so the source no longer blocks half-precision
  inference.

The module is interface-compatible with ``SineGen`` (same constructor arguments,
``forward(f0) -> (sine_waves, uv, noise)``) and touches no trained weights, so it
can be enabled on existing checkpoints.
"""

import torch
from torch import nn


class ExactSineGen(nn.Module):
    """Drop-in replacement for SineGen with integer modular phase accumulation."""

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003, voiced_threshold=0,
                 flag_for_pulse=False, q_bits=48, block_frames=4096):
        super().__init__()
        assert not flag_for_pulse, 'flag_for_pulse is not supported by ExactSineGen'
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = int(upsample_scale)
        self.Q = 1 << q_bits
        self.block_frames = block_frames
        # blockwise cumsum bound: block_frames * Q must fit in int64
        assert block_frames * self.Q < (1 << 62)

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _phase_int(self, f0_frames):
        """f0_frames: [B, T_frames] float. Exact per-sample fundamental phase as
        int64 [B, T_frames * upsample_scale] in [0, Q)."""
        B, Tf = f0_frames.shape
        U, Q = self.upsample_scale, self.Q
        inc = torch.round(f0_frames.double() / self.sampling_rate * Q).long()
        step = (inc * U) % Q
        phi0 = torch.empty_like(step)
        carry = torch.zeros(B, dtype=torch.long, device=step.device)
        for s in range(0, Tf, self.block_frames):
            chunk = step[:, s:s + self.block_frames]
            cs = torch.cumsum(chunk, dim=1)
            phi0[:, s:s + chunk.shape[1]] = (carry.unsqueeze(1) + cs - chunk) % Q
            carry = (carry + cs[:, -1]) % Q
        j = torch.arange(1, U + 1, device=step.device, dtype=torch.long)
        # phi0 < Q = 2**48 and inc * U <= Q * U < 2**57, so the sum fits in int64
        return (phi0.unsqueeze(2) + inc.unsqueeze(2) * j).reshape(B, Tf * U) % Q

    def _f02sine(self, f0_values):
        """Same interface as SineGen._f02sine: f0_values [B, T, dim] with the
        harmonic stack k*f0 at sample rate. Only the fundamental column is used;
        harmonics are derived exactly as (k * phi) mod Q."""
        B, T, D = f0_values.shape
        U, Q = self.upsample_scale, self.Q
        assert T % U == 0, 'input length must be a multiple of upsample_scale'
        phi = self._phase_int(f0_values[:, ::U, 0])
        k = torch.arange(1, D + 1, device=phi.device, dtype=torch.long)
        phk = (phi.unsqueeze(2) * k) % Q            # k <= 9 -> < 2**52, fits int64
        # int64 -> float64 is exact for 48-bit values; the bounded argument makes
        # the sine safe to compute (and cast) at reduced precision
        arg = (phk.double() / Q).to(torch.float32)
        sines = torch.sin(2 * torch.pi * arg)
        if f0_values.dtype.is_floating_point:
            sines = sines.to(f0_values.dtype)
        return sines

    def forward(self, f0):
        """f0: [B, T, 1] at sample rate (nearest-upsampled frame F0).
        Returns (sine_waves, uv, noise) with the same contract as SineGen."""
        fn = f0 * torch.arange(1, self.dim + 1, device=f0.device,
                               dtype=f0.dtype).view(1, 1, -1)
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
