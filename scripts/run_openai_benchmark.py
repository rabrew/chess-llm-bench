#!/usr/bin/env python3
"""Plan by default; --execute explicitly enables billable first-attempt calls."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.openai_benchmark import load_settings, run_benchmark, frozen_manifest_for_run, accept_known_truncations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='results/evaluations.jsonl')
    parser.add_argument('--config',default='config/openai_benchmark.json')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--budget-ledger',required=True)
    parser.add_argument('--phase',choices=('pilot','main'),default='pilot')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--accept-known-truncations',action='store_true')
    args=parser.parse_args()
    settings=load_settings(args.config)
    if args.accept_known_truncations:
        if args.phase!='main' or args.execute:
            parser.error('Maintenance requires --phase main and excludes --execute')
        print(json.dumps(accept_known_truncations(args.source,settings,args.output_dir,args.budget_ledger),indent=2))
        return
    manifest=frozen_manifest_for_run(args.source,settings,args.output_dir)
    print(json.dumps(run_benchmark(manifest,args.output_dir,args.budget_ledger,args.phase,args.execute),indent=2))


if __name__=='__main__':
    main()
