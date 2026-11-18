"""Per-batch tally and geometric-derivative samples from an in-memory OpenMC run.

OpenMC keeps only running sums of geometric derivatives (no sum of squares, so
no derivative uncertainty). Per-batch values are recovered by differencing
those sums after every batch. Nothing here depends on the surface_mesh model.
"""

import numpy as np

DERIVATIVE_SUM_SLOTS = {'batch_gradient': 1, 'batch_datt': 3, 'batch_dJ': 5}

def record_batches(tally_id=1, derivative_id=1, sizer=None):
    """Run every batch in place of ``openmc.lib.run()``; return per-batch values.

    Call inside ``openmc.lib.run_in_memory()``. Returns ``batch_flux`` with
    shape (batches, tally bins) and ``batch_gradient``, ``batch_dJ`` and
    ``batch_datt`` with shape (batches, parameters, tally bins). The batch
    means are checked against OpenMC's own accumulated results. An optional
    ``sizer`` (adaptive_batches.AdaptiveBatches) can end the run early.
    """
    import openmc.lib
    tally = openmc.lib.tallies[tally_id]
    derivative = openmc.lib.geometry_derivatives[derivative_id]

    def running_sums():
        sums = {name: derivative.results[:, :, slot].copy()
                for name, slot in DERIVATIVE_SUM_SLOTS.items()}
        sums['batch_flux'] = tally.results[:, :, 1].reshape(-1).copy()
        return sums

    openmc.lib.simulation_init()
    try:
        start = previous = running_sums()
        rows = {name: [] for name in previous}
        for _ in openmc.lib.iter_batches():
            current = running_sums()
            for name, value in current.items():
                rows[name].append(value - previous[name])
            previous = current
            n = len(rows['batch_flux'])
            if tally.num_realizations != n:
                raise ValueError('Expected one tally realization per batch')
            if sizer is not None and not sizer.after_batch(
                    (current['batch_flux'] - start['batch_flux']) / n,
                    (current['batch_gradient'] - start['batch_gradient']) / n):
                break
        batches = {name: np.asarray(values) for name, values in rows.items()}
        check_batches(batches, tally, derivative)
    finally:
        openmc.lib.simulation_finalize()
    return batches


def check_batches(batches, tally, derivative):
    """Require per-batch values to reproduce OpenMC's means and its tally std. dev."""
    n = tally.num_realizations
    flux = batches['batch_flux']
    if n < 2 or len(flux) != n:
        raise ValueError('Need >=2 realizations, one per recorded batch')
    np.testing.assert_allclose(flux.mean(axis=0), tally.mean.reshape(-1), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(flux.std(axis=0, ddof=1) / np.sqrt(n), tally.std_dev.reshape(-1),
                               rtol=1e-7, atol=1e-14)
    for name, slot in DERIVATIVE_SUM_SLOTS.items():
        expected = derivative.results[:, :, slot] / n
        np.testing.assert_allclose(batches[name].mean(axis=0), expected, rtol=1e-10,
                                   atol=1e-12 * max(np.abs(expected).max(), 1e-300))
    # The engine scores total, attenuation and Jacobian terms in the same call.
    parts = batches['batch_dJ'] + batches['batch_datt']
    np.testing.assert_allclose(batches['batch_gradient'], parts, rtol=1e-9,
                               atol=1e-12 * max(np.abs(parts).max(), 1e-300))
