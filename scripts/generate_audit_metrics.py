#!/usr/bin/env python3
"""Generate CPU audit metrics in an explicitly selected output directory."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import load_results_df, save_metrics
from src.utils import load_config


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/evaluations.jsonl')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--collection-provenance', help='Optional verified historical provenance JSON')
    args = parser.parse_args()
    config = load_config(args.config)
    historical = None
    if args.collection_provenance:
        historical = json.loads(Path(args.collection_provenance).read_text())
        if not isinstance(historical, dict):
            parser.error('collection provenance must be a JSON object')
    snapshot = {key: config[key] for key in ('models', 'benchmark', 'evaluation') if key in config}
    snapshot['source_sha256'] = file_digest(args.results)
    frame = load_results_df(args.results, data_dir=config.get('paths', {}).get('data_dir', 'data'))
    save_metrics(frame, args.output_dir, config_snapshot=snapshot, collection_provenance=historical)
    print(f'Wrote audit metrics for {len(frame)} records to {Path(args.output_dir).resolve()}')


if __name__ == '__main__':
    main()
