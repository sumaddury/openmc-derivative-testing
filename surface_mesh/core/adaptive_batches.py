"""Adaptive batch count from the history of running means (fixed particles per batch).

After batch b, the change in each quantity's running mean m_b (flux bins; the
full gradient vector) updates Welford's recurrence written in running-mean
increments:

    S_b = S_{b-1} + b (b - 1) ||m_b - m_{b-1}||^2,     SE_b^2 = S_b / (b (b - 1)).

Each quantity's relative error r = SE / sqrt(max(||m||^2 - SE^2, 0)) (debiased;
infinite while noise dominates) is compared with its tolerance, and
Q = max_k r_k / tol_k. Relative error falls like 1/sqrt(b), so about b Q^2
batches are needed in total. The test runs only at the end of each planned
chunk, not after every batch, which limits early stops on a lucky batch.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AdaptiveBatchConfig:
    """With ``enabled``, IntRunSettings.batches becomes the batch limit."""

    enabled: bool = False
    flux_rel_tol: float = .01      # relative standard error of the flux tally
    gradient_rel_tol: float = .5   # total relative noise of the gradient vector
    margin: float = .1             # stop only once Q <= 1 / (1 + margin)
    min_batches: int = 10          # warm-up chunk before the first test
    growth_factor: float = 2.      # a chunk adds at most (growth_factor - 1) x batches so far

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('adaptive_batches.enabled must be boolean')
        if isinstance(self.min_batches, bool) or not isinstance(self.min_batches, int) \
                or self.min_batches < 2:
            raise ValueError('adaptive_batches.min_batches must be an integer >= 2')
        for name in ('flux_rel_tol', 'gradient_rel_tol', 'margin', 'growth_factor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'adaptive_batches.{name} must be finite and nonnegative')
        if self.flux_rel_tol == 0 or self.gradient_rel_tol == 0 or self.growth_factor <= 1:
            raise ValueError('Tolerances must be positive and growth_factor must exceed 1')


class RunningError:
    """Relative standard error of a running mean, from the running means alone."""

    def __init__(self):
        self.batches, self.mean, self.sum = 0, None, 0.

    def update(self, mean):
        mean = np.asarray(mean, dtype=float).ravel()
        self.batches += 1
        b = self.batches
        if b > 1:
            self.sum += b * (b - 1) * float(np.sum((mean - self.mean) ** 2))
        self.mean = mean.copy()

    def relative_error(self):
        b = self.batches
        if b < 2:
            return math.inf
        se2 = self.sum / (b * (b - 1))
        signal2 = float(np.sum(self.mean ** 2)) - se2
        return math.sqrt(se2 / signal2) if signal2 > 0 else math.inf


class AdaptiveBatches:
    """Decide after each batch whether to continue; tests only at chunk ends."""

    def __init__(self, config, max_batches):
        if config.min_batches > max_batches:
            raise ValueError('adaptive_batches.min_batches exceeds the batch limit')
        self.config, self.max_batches = config, max_batches
        self.errors = {'flux': RunningError(), 'gradient': RunningError()}
        self.batches, self.next_test, self.checks = 0, config.min_batches, []

    def after_batch(self, flux_mean, gradient_mean):
        """Record running means after a batch; return False to end the simulation."""
        self.errors['flux'].update(flux_mean)
        self.errors['gradient'].update(gradient_mean)
        self.batches += 1
        return self.batches < self.next_test or self._test()

    def _test(self):
        cfg, b = self.config, self.batches
        flux, gradient = (self.errors[name].relative_error() for name in ('flux', 'gradient'))
        q = max(flux / cfg.flux_rel_tol, gradient / cfg.gradient_rel_tol)
        target = q * (1 + cfg.margin)
        if target <= 1:
            status, more = 'converged', 0
        elif b >= self.max_batches:
            status, more = 'batch_limit', 0
        else:
            needed = math.inf if math.isinf(target) else math.ceil(b * (target ** 2 - 1))
            cap = max(1, math.floor((cfg.growth_factor - 1) * b))
            status, more = 'running', int(min(max(needed, 1), cap, self.max_batches - b))
        self.checks.append(dict(batch=b, flux_rel_error=flux, gradient_rel_noise=gradient,
                                q=q, more_batches=more, status=status))
        self.next_test = b + more
        return more > 0

    def arrays(self):
        """Test history as ``adaptive_*`` arrays (one entry per test) for sample.npz."""
        return {f'adaptive_{key}': np.array([check[key] for check in self.checks])
                for key in self.checks[0]} if self.checks else {}
