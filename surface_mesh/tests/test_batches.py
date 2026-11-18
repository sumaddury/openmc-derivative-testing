"""Check per-batch differencing against a fake OpenMC run (no transport)."""

import numpy as np
import openmc.lib
import pytest

from ..core.batches import record_batches


class FakeTally:
    def __init__(self):
        self.results = np.zeros((1, 1, 3))  # current, sum, sum of squares
        self.num_realizations = 0

    @property
    def mean(self):
        return self.results[:, :, 1] / self.num_realizations

    @property
    def std_dev(self):
        n, mean = self.num_realizations, self.mean
        return np.sqrt((self.results[:, :, 2] / n - mean ** 2) / (n - 1))


class FakeDerivative:
    def __init__(self, parameters):
        self.results = np.zeros((parameters, 1, 10))


def fake_run(monkeypatch, flux, dJ, datt, *, gradient_error=0.):
    """Accumulate the given per-batch values as OpenMC does after each batch."""
    tally, derivative = FakeTally(), FakeDerivative(dJ.shape[1])

    def iter_batches():
        for b in range(len(flux)):
            tally.results[0, 0, 1:] += flux[b], flux[b] ** 2
            tally.num_realizations += 1
            derivative.results[:, 0, 1] += dJ[b] + datt[b] + gradient_error
            derivative.results[:, 0, 3] += datt[b]
            derivative.results[:, 0, 5] += dJ[b]
            yield

    monkeypatch.setattr(openmc.lib, 'tallies', {1: tally})
    monkeypatch.setattr(openmc.lib, 'geometry_derivatives', {1: derivative})
    monkeypatch.setattr(openmc.lib, 'simulation_init', lambda: None)
    monkeypatch.setattr(openmc.lib, 'simulation_finalize', lambda: None)
    monkeypatch.setattr(openmc.lib, 'iter_batches', iter_batches)


def random_batches(batches=7, parameters=12):
    rng = np.random.default_rng(1)
    return (rng.uniform(1, 2, batches), rng.normal(size=(batches, parameters)),
            rng.normal(size=(batches, parameters)))


def test_increments_recover_each_batch(monkeypatch):
    flux, dJ, datt = random_batches()
    fake_run(monkeypatch, flux, dJ, datt)
    batches = record_batches()
    np.testing.assert_allclose(batches['batch_flux'][:, 0], flux)
    np.testing.assert_allclose(batches['batch_dJ'][:, :, 0], dJ)
    np.testing.assert_allclose(batches['batch_datt'][:, :, 0], datt)
    np.testing.assert_allclose(batches['batch_gradient'][:, :, 0], dJ + datt)


def test_inconsistent_derivative_terms_are_rejected(monkeypatch):
    flux, dJ, datt = random_batches()
    fake_run(monkeypatch, flux, dJ, datt, gradient_error=1e-3)
    with pytest.raises(AssertionError):
        record_batches()
