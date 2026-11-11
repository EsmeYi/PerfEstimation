"""Verify static artifact files and, optionally, all exported checkpoint hashes."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',action='store_true');ap.add_argument('--raw-data',action='store_true');a=ap.parse_args();entries=json.loads((P/'provenance/package_checksums.json').read_text());count=0
 for rel,v in entries.items():
  p=P/rel
  if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise RuntimeError('Integrity mismatch: '+rel)
  count+=1
 if a.checkpoints:
  for v in [x for x in json.loads((P/'artifacts/manifest.json').read_text())['assets'] if x['group']!='data']:
   p=P/v['path']
   if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise RuntimeError('Checkpoint mismatch: '+str(p))
   count+=1
 if a.raw_data:
  for rel,v in json.loads((P/'provenance/raw_data_manifest.json').read_text()).items():
   p=P/'data/raw'/rel
   if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise RuntimeError('Raw data mismatch: '+str(p))
   count+=1
 print('PASS:',count,'files match recorded sizes and SHA-256 hashes.')
if __name__=='__main__':main()
