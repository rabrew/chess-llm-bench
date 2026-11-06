#!/usr/bin/env python3
"""Create a new blinded offline calibration worksheet from saved snippets."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.human_scoring_pack import build_pack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-results',required=True)
    parser.add_argument('--commercial-root',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--exclusions',required=True,help='Explicit JSON array of previously exposed FENs')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(build_pack(args.local_results,args.commercial_root,args.output_dir,args.exclusions,args.seed),indent=2))


if __name__=='__main__': main()
