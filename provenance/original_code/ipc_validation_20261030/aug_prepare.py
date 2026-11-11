"""Token arrays for sources with the declaration preamble removed (the part that makes a function compilable on its own)."""
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = Path('/proj/berzelius-2026-145/users/x_liryi')
RAW = ROOT / 'pretrained_ipc_20261027/raw/train.jsonl'
TOK = ROOT / 'source_ipc_baseline_20261025/archive/dl4nlp_crossmodal/bpe_tokenizer.json'
BOS, EOS = 2, 3

def strip_preamble(src):
    i = src.find('__attribute__((used))')
    return src[i:] if i >= 0 else None

def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK))
    texts, has = [], []
    with RAW.open() as f:
        for line in f:
            s = strip_preamble(json.loads(line)['source'])
            has.append(s is not None); texts.append(s if s is not None else '')
    out = np.zeros((len(texts), 512), np.uint16)
    for i in range(0, len(texts), 5000):
        for k, e in enumerate(tok.encode_batch(texts[i:i + 5000])):
            ids = [BOS] + e.ids[:510] + [EOS]
            out[i + k, :len(ids)] = ids
        print(i, flush=True)
    np.save(BASE / 'strip_train_tokens.npy', out); np.save(BASE / 'strip_train_has.npy', np.array(has))
    print('done', len(texts), 'strippable', int(np.sum(has)))

if __name__ == '__main__':
    main()
