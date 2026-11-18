"""Check the adaptive batch sizer without running OpenMC transport."""

from dataclasses import replace

import numpy as np
import pytest

from ..core.adaptive_batches import AdaptiveBatchConfig, AdaptiveBatches, RunningError
from ..core.batches import record_batches
from ..core.settings import IntRunSettings
from .test_batches import fake_run


def test_running_error_matches_sample_standard_error():
    x = np.random.default_rng(2).normal(3., 1., size=(30, 5))
    error = RunningError()
    for b in range(1, len(x) + 1):
        error.update(x[:b].mean(axis=0))
    se2 = np.sum(x.var(axis=0, ddof=1) / len(x))
    assert error.relative_error() == pytest.approx(np.sqrt(se2 / (np.sum(x.mean(axis=0) ** 2) - se2)))


def run_sizer(config, max_batches, noise, seed=3):
    """Feed running means of noisy unit-flux, unit-gradient batches until the sizer stops."""
    rng = np.random.default_rng(seed)
    sizer, flux, gradient = AdaptiveBatches(config, max_batches), [], []
    while True:
        flux.append(1 + noise * rng.normal())
        gradient.append(np.ones(4) + noise * rng.normal(size=4))
        if not sizer.after_batch(np.mean(flux), np.mean(gradient, axis=0)):
            return sizer


def test_quiet_run_stops_after_warmup():
    sizer = run_sizer(AdaptiveBatchConfig(enabled=True), max_batches=100, noise=1e-4)
    assert sizer.batches == 10 and [c['status'] for c in sizer.checks] == ['converged']


def test_noisy_run_grows_by_bounded_chunks_until_converged():
    config = AdaptiveBatchConfig(enabled=True, flux_rel_tol=.02, gradient_rel_tol=.05)
    sizer = run_sizer(config, max_batches=10_000, noise=.3)
    tests = [c['batch'] for c in sizer.checks]
    assert sizer.checks[-1]['status'] == 'converged' and tests[0] == 10
    assert all(after <= 2 * before for before, after in zip(tests, tests[1:]))
    assert sizer.checks[-1]['q'] * 1.1 <= 1


def test_batch_limit_ends_run():
    sizer = run_sizer(AdaptiveBatchConfig(enabled=True, gradient_rel_tol=1e-6), 25, noise=.3)
    assert sizer.batches == 25 and sizer.checks[-1]['status'] == 'batch_limit'
    arrays = sizer.arrays()
    assert arrays['adaptive_batch'][-1] == 25 and arrays['adaptive_more_batches'][-1] == 0


def test_record_batches_stops_when_sizer_converges(monkeypatch):
    rng = np.random.default_rng(1)
    flux = 1 + 1e-4 * rng.normal(size=40)
    dJ = 1 + 1e-4 * rng.normal(size=(40, 6))
    fake_run(monkeypatch, flux, dJ, np.zeros_like(dJ))
    sizer = AdaptiveBatches(AdaptiveBatchConfig(enabled=True), max_batches=40)
    batches = record_batches(sizer=sizer)
    assert len(batches['batch_flux']) == 10 and sizer.checks[-1]['status'] == 'converged'


def test_settings_accept_json_and_check_limit():
    cfg = IntRunSettings(batches=50, adaptive_batches={'enabled': True, 'min_batches': 5})
    assert cfg.adaptive_batches == AdaptiveBatchConfig(enabled=True, min_batches=5)
    with pytest.raises(ValueError, match='batch limit'):
        replace(cfg, batches=4)
