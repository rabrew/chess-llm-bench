#!/usr/bin/env python3
"""Prepare, replay/plan/live-grade, or evaluate a frozen candidate semantic judge."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.semantic_explanation_grader import prepare, run, evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    first=commands.add_parser('prepare'); first.add_argument('--cases',required=True); first.add_argument('--config',required=True); first.add_argument('--output-dir',required=True)
    grader=commands.add_parser('grade'); grader.add_argument('--run-dir',required=True); grader.add_argument('--replay'); grader.add_argument('--execute',action='store_true'); grader.add_argument('--ledger')
    evaluator=commands.add_parser('evaluate'); evaluator.add_argument('--run-dir',required=True); evaluator.add_argument('--references',required=True); evaluator.add_argument('--output-file',required=True)
    args=parser.parse_args()
    if args.command=='prepare': result=prepare(args.cases,args.config,args.output_dir)
    elif args.command=='grade': result=run(args.run_dir,replay_file=args.replay,execute=args.execute,ledger_path=args.ledger)
    else: result=evaluate(args.run_dir,args.references,args.output_file)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
