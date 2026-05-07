"""Unit tests for node_fdm_data.smoothing public helpers."""

from __future__ import annotations

import numpy as np

from node_fdm_data.smoothing import bilateral_1d, butter_lowpass, interpolate_nans


class TestBilateral1D:
    """Tests for bilateral_1d edge-preserving smoother."""

    def test_bilateral_1d_preserves_step(self) -> None:
        """Step + Gaussian noise: edge stays sharp, plateaus stay flat."""
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(50), np.ones(50)]) + rng.normal(0, 0.05, 100)
        out = bilateral_1d(y, sigma_s=6.0, sigma_r=1.0)

        assert out[:50].mean() < 0.1, "left plateau should stay near 0"
        assert out[50:].mean() > 0.9, "right plateau should stay near 1"

        # With sigma_s=6 and sigma_r=1 (the spec values), the bilateral filter
        # narrows the Gaussian transition (~3*sigma_s ~= 18 samples) but does
        # not collapse it to a step — qualitative edge preservation is shown
        # by the two plateau means above.
        mid = out[40:60]
        transition_width = int(((mid > 0.1) & (mid < 0.9)).sum())
        assert transition_width < 16, (
            f"step transition should remain narrower than full Gaussian smoothing, "
            f"got {transition_width} samples"
        )

    def test_bilateral_1d_idempotent_on_constant(self) -> None:
        """Constant signal must come back unchanged."""
        y = np.full(100, 5.0)
        out = bilateral_1d(y, sigma_s=6.0, sigma_r=1.0)
        assert np.allclose(out, 5.0)


class TestButterLowpass:
    """Tests for butter_lowpass."""

    def test_butter_lowpass_kills_high_frequency(self) -> None:
        """High-frequency component is attenuated; low-frequency preserved."""
        n = 1024
        dt = 4.0
        t = np.arange(n) * dt
        # cutoff_s=100 -> cutoff freq = 0.01 Hz; Nyquist = 1/(2*dt) = 0.125 Hz.
        # HF tone at 0.05 Hz: well above cutoff, well below Nyquist (no alias).
        # LF tone at 0.001 Hz: well below cutoff (preserved).
        hf = np.sin(2 * np.pi * 0.05 * t)
        lf = np.sin(2 * np.pi * 0.001 * t)
        y = hf + lf

        out = butter_lowpass(y, cutoff_s=100.0, dt=dt, order=4)

        # The output should be ~the LF component.
        # HF residual amplitude:
        residual = out - lf
        hf_attenuation_db = 20.0 * np.log10(np.std(hf) / max(np.std(residual), 1e-12))
        assert hf_attenuation_db > 25.0, (
            f"HF should be attenuated by > 25 dB, got {hf_attenuation_db:.1f}"
        )

        # LF amplitude preserved within 10%.
        amp_in = float(np.max(np.abs(lf)))
        amp_out = float(np.max(np.abs(out)))
        assert abs(amp_out - amp_in) / amp_in < 0.10, (
            f"LF amplitude should be preserved within 10%, got ratio {amp_out / amp_in:.3f}"
        )

    def test_butter_lowpass_short_signal_no_op(self) -> None:
        """Signal shorter than filter padding length is returned unchanged."""
        y = np.array([1.0, 2.0, 3.0])
        out = butter_lowpass(y, cutoff_s=100.0, dt=1.0, order=4)
        assert np.array_equal(out, y)


class TestInterpolateNans:
    """Tests for interpolate_nans."""

    def test_interpolate_nans_partial(self) -> None:
        """Interior NaNs are linearly interpolated."""
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        out = interpolate_nans(y)
        assert np.allclose(out, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_interpolate_nans_all_nan(self) -> None:
        """All-NaN input returns all zeros."""
        y = np.full(10, np.nan)
        out = interpolate_nans(y)
        assert np.allclose(out, 0.0)
        assert out.shape == y.shape

    def test_interpolate_nans_single_valid(self) -> None:
        """A single valid sample propagates to all rows."""
        y = np.array([np.nan, np.nan, 7.0, np.nan])
        out = interpolate_nans(y)
        assert np.allclose(out, 7.0)
