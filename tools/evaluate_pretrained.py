"""Compatibility entry point for the FINAL fixed9762-update pretrained models."""
from pathlib import Path
import argparse,subprocess,sys
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['codebert','unixcoder'],required=True);ap.add_argument('--condition',choices=['ipc','joint'],default='ipc');ap.add_argument('--seed',type=int,choices=[42,43,44],default=42);ap.add_argument('--scope',choices=['original','full','filtered'],default='original');ap.add_argument('--output',type=Path);a=ap.parse_args()
args=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'--model',a.model+'_'+a.condition,'--seed',str(a.seed),'--scope',a.scope]
if a.output:args+=['--output',str(a.output)]
subprocess.run(args,check=True)
