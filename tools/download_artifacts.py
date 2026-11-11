"""Download optional public Release assets, check SHA256, and install safely."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,tarfile,time,urllib.request
P=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fetch(item,base,cache):
 out=cache/item['asset'];partial=out.with_suffix(out.suffix+'.part')
 if out.exists() and sha(out)==item['asset_sha256']:return out
 for attempt in range(3):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   request=urllib.request.Request(base+'/'+item['asset'],headers={'User-Agent':'PerfEstimation-replication',**({'Range':f'bytes={offset}-'} if offset else {})})
   with urllib.request.urlopen(request,timeout=120) as response,partial.open('ab' if offset and response.status==206 else 'wb') as target:shutil.copyfileobj(response,target,1024*1024)
   if sha(partial)!=item['asset_sha256']:partial.unlink();raise ValueError('Checksum mismatch: '+item['asset'])
   partial.replace(out);return out
  except Exception:
   if attempt==2:raise
   time.sleep(2**attempt)
def install(item,asset):
 if item['group']=='data':
  allowed={f'data/raw/{name}.jsonl' for name in ['train','val','test']}
  with tarfile.open(asset,'r:gz') as tar:
   members=tar.getmembers()
   if len(members)!=3 or {x.name for x in members}!=allowed or not all(x.isfile() for x in members):raise ValueError('Unexpected archive members; refusing installation.')
   for member in members:
    out=P/member.name;out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp')
    with tar.extractfile(member) as f,tmp.open('wb') as g:shutil.copyfileobj(f,g,8*1024*1024)
    tmp.replace(out)
  return
 out=(P/item['path']).resolve()
 if P.resolve() not in out.parents:raise ValueError('Checkpoint path escapes repository.')
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists() and sha(out)==item['sha256']:return
 tmp=out.with_suffix('.tmp')
 with gzip.open(asset,'rb') as f,tmp.open('wb') as g:shutil.copyfileobj(f,g,8*1024*1024)
 if tmp.stat().st_size!=item['bytes'] or sha(tmp)!=item['sha256']:
  tmp.unlink();raise ValueError('Decompressed checkpoint checksum mismatch: '+str(out))
 tmp.replace(out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--list',action='store_true');ap.add_argument('--group',choices=['data','compact','pretrained','all']);ap.add_argument('--model');ap.add_argument('--seed',type=int,choices=[42,43,44]);a=ap.parse_args()
 manifest=json.loads((P/'artifacts/manifest.json').read_text());items=manifest['assets']
 if a.group and a.group!='all':items=[x for x in items if x['group']==a.group]
 if a.model:items=[x for x in items if x.get('model')==a.model]
 if a.seed:items=[x for x in items if x.get('seed')==a.seed]
 if not items:raise SystemExit('No matching assets. Use --list.')
 if a.list or not(a.group or a.model):
  for x in items:print(x['asset'],f'{x["asset_bytes"]/2**20:.1f} MiB',x.get('model','raw data'))
  return
 cache=P/'downloads';cache.mkdir(exist_ok=True)
 for item in items:
  print('Downloading/verifying',item['asset'],flush=True);asset=fetch(item,manifest['base_url'],cache);install(item,asset);print('INSTALLED',item.get('path','data/raw'),flush=True)
if __name__=='__main__':main()
