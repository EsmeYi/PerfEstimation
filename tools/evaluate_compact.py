"""Evaluate an exported final compact checkpoint with the fixed 5k Ridge readout."""
import argparse,csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1];E=P/'experiments';C=E/'source_ipc_baseline_20261025'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--modality',choices=['source','assembly'],default='source');ap.add_argument('--device',default='cuda');ap.add_argument('--output',type=Path,default=P/'generated/checkpoint_evaluation');a=ap.parse_args()
 sys.path.insert(0,str(C/'archive/Experiment'));spec=importlib.util.spec_from_file_location('encoder',C/'archive/Experiment/final_train_seeded.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);key='src_enc' if a.modality=='source' else 'asm_enc'
 if key not in state:raise ValueError(f'{a.checkpoint} does not contain {key}; use source for source-only models.')
 vocab=state[key]['tok_emb.weight'].shape[0];net=m.Encoder(vocab).to(a.device);net.load_state_dict(state[key]);net.eval();torch.set_num_threads(4)
 if a.modality=='source':tr=C/'source_only_data/train_tokens.npy';te=C/'source_only_data/test_tokens.npy'
 else:
  folder='ia_fixed_joint_ablation_20261026' if vocab==254 else 'bpe_joint_ablation_20261025';suffix='ia_fixed' if vocab==254 else 'bpe';tr=E/f'{folder}/assembly_data/train_asm_{suffix}.npy';te=E/f'{folder}/assembly_data/test_asm_{suffix}.npy'
 idx=np.load(C/'source_only_data/probe_train_indices.npy');train=np.load(tr,mmap_mode='r');test=np.load(te,mmap_mode='r');labels=np.load(C/'source_only_data/train_ipc.npy');ref=np.load(C/'probe_embeddings_seed42.npz',allow_pickle=False)
 def encode(data,ix):
  zs=[]
  with torch.inference_mode():
   for off in range(0,len(ix),16):zs.append(net(torch.tensor(np.array(data[ix[off:off+16]],dtype=np.int64),device=a.device)).cpu().numpy())
  return np.concatenate(zs).astype(np.float64)
 x=encode(train,idx);y=np.log1p(labels[idx].astype(np.float64));xm=x.mean(0);ym=y.mean();xc=x-xm;coef=np.linalg.solve(xc.T@xc+np.eye(128),xc.T@(y-ym));z=encode(test,np.arange(len(test)));pred=(z-xm)@coef+ym
 actual=ref['test_ipc'].astype(float);target=np.log1p(actual);r2=float(1-np.square(target-pred).sum()/np.square(target-target.mean()).sum());mae=float(abs(actual-np.expm1(pred)).mean())
 a.output.mkdir(exist_ok=True,parents=True);np.savez(a.output/'readout.npz',coef=coef,x_mean=xm,y_mean=ym)
 with (a.output/'test_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','ipc_llvm_mca','probe_log1p','probe_ipc']);w.writerows(zip(ref['test_ids'],actual,pred,np.expm1(pred)))
 result={'r2_log1p':r2,'mae_ipc':mae,'n':len(test),'checkpoint':str(a.checkpoint),'modality':a.modality,'precision':'float32 embeddings, float64 Ridge solve','ridge_alpha':1,'n_probe_train':len(idx)};(a.output/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
