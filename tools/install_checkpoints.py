"""Link evaluation-only exports into historical paths; refuses to overwrite any run checkpoint."""
import argparse,json
from pathlib import Path
P=Path(__file__).resolve().parents[1];E=P/'experiments'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=P/'checkpoints');a=ap.parse_args()
 for m in ['IA','BPE','source_only','alignment_only','codebert','unixcoder']:
  for seed in [42,43,44]:
   src=a.source/f'{m}_seed{seed}.pt'
   if not src.exists():continue
   if m in ['IA','BPE']:
    dst=E/f'ia_bpe_validation_search_20261026/trials/trial_{7 if m=="IA" else 15:02d}/checkpoint.pt' if seed==42 else E/f'ia_bpe_tuned_multiseed_20261027/runs/{m}_seed{seed}/checkpoint.pt'
   elif m in ['source_only','alignment_only']:dst=E/f'supervision_ablations_20261027/runs/{m}_seed{seed}/checkpoint.pt'
   else:dst=E/f'pretrained_ipc_30min_20261028/runs/{m}_lr5e-05_seed{seed}/checkpoint.pt'
   dst.parent.mkdir(exist_ok=True,parents=True)
   if dst.exists():
    if dst.resolve()!=src.resolve():raise FileExistsError(f'Refusing to replace {dst}')
   else:dst.symlink_to(src.resolve())
 print('Installed evaluation-only checkpoint links. Do not resume training in these run directories; use a fresh package copy for retraining.')
if __name__=='__main__':main()
