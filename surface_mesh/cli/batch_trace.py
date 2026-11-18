"""Evaluate one mesh (no optimization) and save its per-batch tally and vertex gradients.

The output directory receives sample.npz (the usual evaluation sample plus
batch_flux, batch_gradient, batch_dJ and batch_datt), settings.json and
transport.log. Geometry images are skipped. Batch recording needs a fixed
batch count, i.e. trigger_max_factor = 1 (the default).
"""

import argparse
from dataclasses import replace
from pathlib import Path

from ..core.evaluate import evaluate
from ..core.settings import IntRunSettings, load_run_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the results')
    parser.add_argument('--run-config', type=Path, help='int_run JSON (default: IntRunSettings defaults)')
    parser.add_argument('--model', choices=['sphere', 'box'])
    for name in ('particles', 'batches', 'threads', 'seed'):
        parser.add_argument(f'--{name}', type=int)
    parser.add_argument('--adaptive', action='store_true',
                        help='Choose the batch count at runtime (adaptive_batches); --batches is the limit')
    args = parser.parse_args(argv)
    cfg = load_run_config(args.run_config) if args.run_config else IntRunSettings()
    updates = {key: getattr(args, arg) for key, arg in
               [('model_variant', 'model'), ('particles', 'particles'), ('batches', 'batches'),
                ('threads', 'threads'), ('seed', 'seed')]
               if getattr(args, arg) is not None}
    if args.adaptive:
        updates['adaptive_batches'] = replace(cfg.adaptive_batches, enabled=True)
    cfg = replace(cfg, **updates, record_batches=True,
                  save_geometry_images=False, save_solid_geometry_images=False)
    sample = evaluate(args.mesh, args.output, cfg, cfg.seed)
    print(f"Saved {len(sample['batch_flux'])} batches x {sample['batch_gradient'].shape[1]} vertices "
          f"to {args.output / 'sample.npz'}; tally {sample['tally_mean']:.6g} "
          f"+/- {sample['tally_std_dev']:.2g}")
    if 'adaptive_status' in sample:
        print(f"Adaptive batches: {sample['adaptive_status'][-1]} after tests at batches "
              f"{sample['adaptive_batch'].tolist()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
