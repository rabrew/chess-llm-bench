#!/usr/bin/env python3
"""Audit saved explanations and human calibration offline; no model calls."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.explanation_audit import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('local-results','commercial-root','pack-root','ratings','original-ratings','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--config',default='config/config.yaml')
    args=parser.parse_args()
    with open(args.config) as stream: models=yaml.safe_load(stream)['models']
    print(json.dumps(audit(args.local_results,args.commercial_root,args.pack_root,args.ratings,args.original_ratings,
                           args.output_dir,local_models=models),indent=2))


if __name__=='__main__': main()
