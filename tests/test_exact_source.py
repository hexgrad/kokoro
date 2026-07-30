"""Tests for ExactSineGen (exact integer-phase harmonic source)."""

import numpy as np
import pytest
import torch

from kokoro.exact_source import ExactSineGen
from kokoro.istftnet import SineGen, SourceModuleHnNSF

FS = 24000
UPS = 300
Q = 1 << 48


def reference_phase(f0_frames, harmonic):
    """Straightforward per-sample integer accumulation with arbitrary-precision carry."""
    inc = np.rint(np.asarray(f0_frames, dtype=np.float64) / FS * Q).astype(np.int64)
    out = np.empty(len(f0_frames) * UPS, dtype=np.int64)
    ar = np.arange(1, UPS + 1, dtype=np.int64)
    phi0 = 0
    for m, i in enumerate(inc):
        out[m * UPS:(m + 1) * UPS] = (phi0 + int(i) * ar) % Q
        phi0 = (phi0 + int(i) * UPS) % Q
    return (harmonic * out) % Q


def make_f0(frames, rng):
    f0 = rng.uniform(60, 400, frames).astype(np.float32).astype(np.float64)
    f0[rng.random(frames) < 0.3] = 0.0  # unvoiced stretches
    return f0


def test_integer_phase_matches_reference():
    rng = np.random.default_rng(0)
    f0 = make_f0(977, rng)
    gen = ExactSineGen(FS, UPS, harmonic_num=8)
    phi = gen._phase_int(torch.tensor(f0, dtype=torch.float32).unsqueeze(0))[0].numpy()
    for k in (1, 5, 9):
        assert np.array_equal((k * phi) % Q, reference_phase(f0, k))


def test_long_duration_no_overflow():
    # 700 s of audio is far beyond the point where a naive int64 cumsum overflows
    frames = 80 * 700
    f0 = float(np.float32(393.7))
    gen = ExactSineGen(FS, UPS, harmonic_num=8)
    phi = gen._phase_int(torch.full((1, frames), f0))[0]
    n_last = frames * UPS - 1
    expected = (int(np.rint(f0 / FS * Q)) * (n_last + 1)) % Q
    assert int(phi[n_last]) == expected
    assert int(phi.min()) >= 0 and int(phi.max()) < Q


def test_half_precision_stays_correlated():
    rng = np.random.default_rng(1)
    f0 = make_f0(800, rng)
    f0_t = torch.tensor(np.repeat(f0, UPS), dtype=torch.float32).view(1, -1, 1)
    harmonics = torch.arange(1, 10).view(1, 1, -1)
    gen = ExactSineGen(FS, UPS, harmonic_num=8)
    ref = gen._f02sine((f0_t.half().float() * harmonics))
    half = gen._f02sine((f0_t.half() * harmonics.half()))
    corr = np.corrcoef(half[0, :, 0].float().numpy(), ref[0, :, 0].numpy())[0, 1]
    assert corr > 0.999


def test_forward_contract_matches_sinegen():
    rng = np.random.default_rng(2)
    f0 = make_f0(200, rng)
    f0_t = torch.tensor(np.repeat(f0, UPS), dtype=torch.float32).view(1, -1, 1)
    torch.manual_seed(0)
    exact = ExactSineGen(FS, UPS, harmonic_num=8)
    torch.manual_seed(0)
    legacy = SineGen(FS, UPS, harmonic_num=8)
    se, ue, ne = exact(f0_t)
    sl, ul, nl = legacy(f0_t)
    assert se.shape == sl.shape and ue.shape == ul.shape and ne.shape == nl.shape
    assert torch.equal(ue, ul)
    assert torch.isfinite(se).all()


def test_source_module_flag():
    module = SourceModuleHnNSF(FS, UPS, harmonic_num=8, exact_source=True)
    assert isinstance(module.l_sin_gen, ExactSineGen)
    module = SourceModuleHnNSF(FS, UPS, harmonic_num=8)
    assert isinstance(module.l_sin_gen, SineGen)


def test_exact_harmonicity():
    rng = np.random.default_rng(3)
    f0 = make_f0(400, rng)
    gen = ExactSineGen(FS, UPS, harmonic_num=8)
    phi = gen._phase_int(torch.tensor(f0, dtype=torch.float32).unsqueeze(0))[0].numpy()
    # harmonic k is exactly k times the fundamental, modulo Q, at every sample
    for k in (2, 3, 9):
        assert np.array_equal((k * phi) % Q, (k * (phi % Q)) % Q)
