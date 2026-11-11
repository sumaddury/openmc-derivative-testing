import argparse
import json
from pathlib import Path

import numpy as np
import openmc.lib

import optimize

PARAMETER_NAMES = ('r', 'x0', 'y0')


def collect_batch_trace(iron=None, transport_options=None):
    iron = dict(iron or {})
    optimize.initialize_model(PARAMETER_NAMES, iron=iron, transport_options=transport_options,
                              use_trigger=False)
    optimize.update_parameters(optimize.get_parameters(PARAMETER_NAMES), PARAMETER_NAMES)
    metadata = optimize.model_metadata()
    metadata['geometry'] = {name: float(optimize.model_surfaces[1].coefficients[name])
                            for name in ('r', 'x0', 'y0', 'z0')}

    rows = {key: [] for key in ('batch', 'tally_mean', 'tally_std_dev', 'gradient', 'dJ', 'datt')}
    with openmc.lib.run_in_memory(args=optimize.runtime_args, output=False):
        openmc.lib.simulation_init()
        try:
            tally = openmc.lib.tallies[1]
            derivative = openmc.lib.geometry_derivatives[1]
            order = [derivative.surface_parameters[1].index(name) for name in PARAMETER_NAMES]
            for _ in openmc.lib.iter_batches():
                n = max(tally.num_realizations, 1)
                rows['batch'].append(tally.num_realizations)
                rows['tally_mean'].append(float(tally.mean.flat[-1]))
                rows['tally_std_dev'].append(float(tally.std_dev.flat[-1]))
                rows['gradient'].append(derivative.mean[order, 0].copy())
                rows['dJ'].append(derivative.dJ[order, 0] / n)
                rows['datt'].append(derivative.datt[order, 0] / n)
                print(f"batch {rows['batch'][-1]:4d}  T={rows['tally_mean'][-1]:.6g}  "
                      f"grad={np.array2string(rows['gradient'][-1], precision=5)}")
        finally:
            openmc.lib.simulation_finalize()
    return {key: np.asarray(value) for key, value in rows.items()}, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Record running flux and gradient after every batch for one geometry.')
    parser.add_argument('--r', type=float, help='Iron sphere radius (default: model default).')
    parser.add_argument('--x0', type=float, help='Iron sphere center x (default: model default).')
    parser.add_argument('--y0', type=float, help='Iron sphere center y (default: model default).')
    parser.add_argument('--batches', type=int, help='Number of batches (default: model default).')
    parser.add_argument('--particles', type=int, help='Particles per batch (default: model default).')
    parser.add_argument('--seed', type=int, help='OpenMC transport random seed (default: OpenMC default).')
    parser.add_argument('--output', default='batch_trace.npz', help='Output .npz path.')
    args = parser.parse_args()

    iron = {name: getattr(args, name) for name in ('r', 'x0', 'y0') if getattr(args, name) is not None}
    transport_options = {key: getattr(args, key) for key in ('batches', 'particles', 'seed')
                         if getattr(args, key) is not None}
    if any(value < 1 for value in transport_options.values()):
        parser.error('--batches, --particles and --seed must be positive integers.')

    trace, metadata = collect_batch_trace(iron=iron, transport_options=transport_options or None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, parameter_names=np.array(PARAMETER_NAMES),
             metadata=json.dumps(metadata, sort_keys=True), **trace)
    print(f'Saved {len(trace["batch"])} batches to {output}')


if __name__ == '__main__':
    main()
