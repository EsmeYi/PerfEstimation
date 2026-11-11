"""Acquire frozen public pretrained initializations and build native tokenizer caches."""
import argparse,importlib.util,sys,json
from pathlib import Path
P=Path(__file__).resolve().parents[1];E=P/'experiments'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['codebert','unixcoder','all'],default='all');ap.add_argument('--tokenize-only',action='store_true',help='Offline tokenization using archived tokenizer metadata; do not download initial weights.');a=ap.parse_args()
 # Import only the historical preparation function, never the obsolete fixed-step training entry point.
 path=E/'pretrained_ipc_20261027/prepare_original.py';spec=importlib.util.spec_from_file_location('pretrained_prepare',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for name in (['codebert','unixcoder'] if a.model=='all' else [a.model]):m.prepare(name,tokenize_only=a.tokenize_only)
if __name__=='__main__':main()
