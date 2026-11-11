import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np

import collect_batch_trace
import optimize
from config import OPTIMIZATION_PARAMETER_NAMES

REFERENCE_BATCHES = 100
REFERENCE_PARTICLES = 1_000_000
REFERENCE_SEED = 123456789

COLUMNS = [
    'batches', 'particles', 'seed',
    'T_ref', 'T_ref_std_dev',
    'r', 'x0', 'y0',
    'grad_r', 'grad_x0', 'grad_y0',
    'evaluations', 'iterations', 'neutrons', 'optimization_seconds',
    'success', 'message', 'error',
]


def run_cell(batches, particles, seed, reference, history_path):
    row = {'batches': batches, 'particles': particles, 'seed': seed}
    start = time.perf_counter()
    try:
        result = optimize.optimize_parameter(
            output_path=history_path, method='local',
            transport_options={'batches': batches, 'particles': particles, 'seed': seed},
            use_trigger=False, capture_images=False,
        )
    except Exception as error:
        row['optimization_seconds'] = time.perf_counter() - start
        row['error'] = f'optimization: {type(error).__name__}: {error}'
        return row
    row['optimization_seconds'] = time.perf_counter() - start
    final = {name: float(value) for name, value in zip(OPTIMIZATION_PARAMETER_NAMES, result.x)}
    row.update(final)
    row['evaluations'] = int(result.nfev)
    row['iterations'] = int(result.nit)
    row['neutrons'] = int(result.nfev) * batches * particles
    row['success'] = bool(result.success)
    row['message'] = str(result.message)
    try:
        trace, _ = collect_batch_trace.collect_batch_trace(iron=final, transport_options=reference)
    except Exception as error:
        row['error'] = f'reference: {type(error).__name__}: {error}'
        return row
    row['T_ref'] = float(trace['tally_mean'][-1])
    row['T_ref_std_dev'] = float(trace['tally_std_dev'][-1])
    for name, value in zip(('grad_r', 'grad_x0', 'grad_y0'), trace['gradient'][-1]):
        row[name] = float(value)
    return row


def completed_cells(results_path):
    if not results_path.exists():
        return set()
    with open(results_path, newline='') as stream:
        return {(int(r['batches']), int(r['particles']), int(r['seed']))
                for r in csv.DictReader(stream) if not r['error']}


def append_row(results_path, row):
    new_file = not results_path.exists()
    with open(results_path, 'a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({column: row.get(column, '') for column in COLUMNS})


def main():
    parser = argparse.ArgumentParser(
        description='Optimize at each (batches, particles, seed) and score the final '
                    'geometry with a fixed-budget reference run on a separate seed.')
    parser.add_argument('--batches', type=int, nargs='+', required=True)
    parser.add_argument('--particles', type=int, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True,
                        help='Directory for results.csv and per-cell histories.')
    parser.add_argument('--ref-batches', type=int, default=REFERENCE_BATCHES)
    parser.add_argument('--ref-particles', type=int, default=REFERENCE_PARTICLES)
    parser.add_argument('--ref-seed', type=int, default=REFERENCE_SEED)
    args = parser.parse_args()

    values = args.batches + args.particles + args.seeds + [args.ref_batches, args.ref_particles, args.ref_seed]
    if any(value < 1 for value in values):
        parser.error('All batch counts, particle counts and seeds must be positive integers.')
    if args.ref_seed in args.seeds:
        parser.error('--ref-seed must differ from every study seed.')

    reference = {'batches': args.ref_batches, 'particles': args.ref_particles, 'seed': args.ref_seed}
    output_dir = Path(args.output_dir)
    (output_dir / 'histories').mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.csv'
    done = completed_cells(results_path)

    cells = list(itertools.product(args.batches, args.particles, args.seeds))
    for index, (batches, particles, seed) in enumerate(cells, start=1):
        label = f'batches={batches} particles={particles} seed={seed}'
        if (batches, particles, seed) in done:
            print(f'[{index}/{len(cells)}] skipping {label}: already done')
            continue
        print(f'[{index}/{len(cells)}] {label}')
        history_path = output_dir / 'histories' / f'b{batches}_p{particles}_s{seed}.npy'
        row = run_cell(batches, particles, seed, reference, history_path)
        append_row(results_path, row)
        if row.get('error'):
            print(f'  failed: {row["error"]}')
        else:
            print(f'  T_ref={row["T_ref"]:.6g} +/- {row["T_ref_std_dev"]:.2g}  '
                  f'r={row["r"]:.4g} x0={row["x0"]:.4g} y0={row["y0"]:.4g}  '
                  f'evaluations={row["evaluations"]}')


if __name__ == '__main__':
    main()
