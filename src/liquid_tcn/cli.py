"""prepare / train / evaluate / export command line."""
import argparse
from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .ported.storage import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--config'); prep.add_argument('--output', required=True)
    prep.add_argument('--root', help='Discover completed attempts recursively under this collection root')
    fit = sub.add_parser('train'); fit.add_argument('--data'); fit.add_argument('--output')
    fit.add_argument('--config', required=True); fit.add_argument('--resume'); fit.add_argument('--max-updates', type=int)
    fit.add_argument('--device'); fit.add_argument('--dry-run', action='store_true')
    ev = sub.add_parser('evaluate'); ev.add_argument('--data', required=True); ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--split', choices=('train','validation','test'), default='test'); ev.add_argument('--output', required=True)
    ev.add_argument('--device', default='cpu'); ev.add_argument('--batch-size', type=int, default=16)
    ev.add_argument('--num-workers', type=int, default=0); ev.add_argument('--max-batches', type=int)
    ex = sub.add_parser('export'); ex.add_argument('--checkpoint', required=True); ex.add_argument('--output', required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        from .preprocess import prepare
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        if args.root:
            config['collection_root'] = args.root
        if not config.get('collection_root') and not config.get('sources'):
            p.error('prepare needs --root or a config with collection_root/sources')
        result = prepare(config, args.output)
        print(json.dumps(dict(windows=result['windows'], profile=result['profile'],
                              discovery=result['discovery']['counts'] if result['discovery'] else None), indent=2)); return
    if args.command == 'train':
        from .train import train, DEFAULTS
        selected = json.loads(Path(args.config).read_text())
        configuration = dict(selected.get('training', selected))
        if args.device:
            configuration['device'] = args.device
        data = args.data or selected.get('data')
        output = args.output or selected.get('output')
        if not data or not output:
            p.error('Training needs data and output paths in JSON or --data/--output')
        output = output.replace('{timestamp}', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        resume = args.resume or selected.get('resume')
        maximum = args.max_updates if args.max_updates is not None else selected.get('max_updates')
        if args.dry_run:
            print(json.dumps(dict(data=data, output=output, training=dict(DEFAULTS, **configuration),
                                  model=selected.get('model'), resume=resume, max_updates=maximum), indent=2))
            return
        print(json.dumps(train(data, output, configuration, resume, maximum,
                               selected.get('model'), selected), indent=2)); return
    from .checkpoint import load_checkpoint, atomic_save
    saved = load_checkpoint(args.checkpoint)
    if args.command == 'export':
        from .metadata import PROPERTY_FIELDS
        artifact = dict(format_version=1, model=saved['model'], model_config=saved['model_config'],
            normalization=saved['normalization'], data_manifest_sha256=saved['data_manifest_sha256'],
            output_order=['force','moment','r_B','v_B'], output_units=['N','N m','m','m/s'],
            input_order=['g_B','u_O_B','omega_B','held_a_B','held_alpha_B'],
            property_fields=PROPERTY_FIELDS, sample_seconds=.005,
            interface='liquid_tcn.inference.load_export; CachedInference.step returns [B,12] and warmup validity',
            convention='completed 5 ms interval, endpoint base frame, moment about fixed endpoint base origin',
            data_provenance=saved['data_provenance'])
        atomic_save(args.output, artifact); print(args.output); return
    from .model import LiquidTCN, ModelConfig
    from .normalization import Normalization
    from .dataset import WindowDataset, make_loader
    from .evaluate import evaluate
    torch.set_num_threads(4)
    if digest(Path(args.data)/'manifest.json') != saved['data_manifest_sha256']:
        raise ValueError('Evaluation data differs from checkpoint provenance')
    dataset = WindowDataset(args.data, args.split, verify=True)
    loader = make_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = LiquidTCN(ModelConfig(**saved['model_config'])).to(args.device)
    model.load_state_dict(saved['model'])
    result = evaluate(model, loader, Normalization(saved['normalization']), args.device, args.max_batches)
    result.update(checkpoint_sha256=digest(args.checkpoint), data_manifest_sha256=saved['data_manifest_sha256'], split=args.split)
    atomic_json(args.output, result)
    print(json.dumps(dict(status=result['status'], selection_metric=result['selection_metric'], coverage=result['coverage']), indent=2))


if __name__ == '__main__':
    main()
