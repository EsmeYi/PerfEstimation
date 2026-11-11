"""Fast checks of vocabulary, architecture, and package imports; no training or network."""
from pathlib import Path
import importlib.util,json,sys,torch,numpy as np
from tokenizers import Tokenizer
P=Path(__file__).resolve().parents[1];E=P/'experiments'
A=E/'ia_fixed_joint_ablation_20261026';sys.path.insert(0,str(A));from test_tokenizer import check
check(json.loads((A/'tokenizer_spec.json').read_text()))
C=E/'source_ipc_baseline_20261025/archive/Experiment';sys.path.insert(0,str(C));spec=importlib.util.spec_from_file_location('arch',C/'final_train_seeded.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1);net=m.Encoder(16384);assert sum(p.numel() for p in net.parameters())==7517696
net.eval()
with torch.inference_mode():z=net(torch.tensor([[2,10,3,0],[2,20,21,3]]))
assert z.shape==(2,128) and torch.isfinite(z).all()
tk=Tokenizer.from_file(str(P/'data/tokenizers/bpe_tokenizer.json'));assert tk.get_vocab_size()==16384
ix=np.load(P/'data/splits/probe_train_indices.npy');assert len(ix)==len(set(ix.tolist()))==5000
assert len(json.loads((P/'data/splits/test_ids.json').read_text()))==2000
print('PASS: tokenizer checks, encoder parameter count/forward, frozen BPE vocabulary and split identities.')
