"""Minimal fix: extract normalized operand tokens inside %, (), and commas.

Original vocabulary, register patterns, memory/immediate regexes, opcode rules,
source tokenization and sequence cap are preserved. Inline comments are ignored
so register-like text in comments cannot become operand tokens. No new operand
features, register identities, instruction-dependency model, or vocabulary IDs.
"""
from functools import lru_cache
import re

class FixedIATokenizer:
    def __init__(self, specification):
        self.vocab=specification['asm_vocab']
        self.registers=[(re.compile(p),token) for p,token in specification['register_patterns']]
        self.memory=re.compile(specification['memory_pattern'])
        self.immediate=re.compile(specification['immediate_pattern'])
        self.operand_tokens=set(specification['operand_tokens'])
        self.abstract=re.compile(r'\[[A-Z0-9]+\]')

    @lru_cache(maxsize=65536)
    def encode_line(self,line):
        line=line.split('#',1)[0].strip()
        if not line or line.startswith('.') or line.endswith(':'):
            return ()
        parts=line.split(None,1)
        opcode=f"[{parts[0].lower().rstrip(',').upper()}]"
        ids=[self.vocab.get(opcode,1)]
        if len(parts)>1:
            operands=parts[1]
            for pattern,token in self.registers:
                operands=pattern.sub(token,operands)
            operands=self.memory.sub('[MEM](',operands)
            operands=self.immediate.sub('[IMM]',operands)
            ids.extend(self.vocab[t] for t in self.abstract.findall(operands) if t in self.operand_tokens)
        return tuple(ids)

    def encode(self,text,max_len=512):
        ids=[2]
        for line in text.splitlines():
            ids.extend(self.encode_line(line))
            if len(ids)>=max_len-1:break
        return ids[:max_len-1]+[3]
