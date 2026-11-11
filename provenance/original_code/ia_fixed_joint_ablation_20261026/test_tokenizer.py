"""Regression checks against real AT&T operand syntax and comment handling."""
import json
from pathlib import Path
from fixed_tokenizer import FixedIATokenizer

def check(spec):
    tokenizer=FixedIATokenizer(spec)
    reverse={v:k for k,v in tokenizer.vocab.items()}
    cases={
        'addq %rax, %rbx':['[ADDQ]','[GPR64]','[GPR64]'],
        'movq 8(%rsp), %rax':['[MOVQ]','[MEM]','[GPR64]','[GPR64]'],
        'leaq 8(%rax,%rcx,4), %rdx':['[LEAQ]','[MEM]','[GPR64]','[GPR64]','[GPR64]'],
        'addq $8, %rsp':['[ADDQ]','[IMM]','[GPR64]'],
        'movl %eax, %r8d':['[MOVL]','[GPR32]','[GPR32]'],
        'vaddps %ymm0, %ymm1, %ymm2':['[VADDPS]','[SIMD256]','[SIMD256]','[SIMD256]'],
        'movq symbol(%rip), %rax':['[MOVQ]','[MEM]','[GPR64]'],
        'addq %rax, %rbx # %xmm0 is a comment':['[ADDQ]','[GPR64]','[GPR64]'],
        'callq external_symbol':['[CALLQ]'],
        '.cfi_def_cfa_offset 16':[], '.Llabel:':[], '# comment %rax':[],
    }
    for line,expected in cases.items():
        actual=[reverse[x] for x in tokenizer.encode_line(line)]
        assert actual==expected,(line,actual,expected)
    assert tokenizer.encode('addq %rax, %rbx',max_len=4)==[2,tokenizer.vocab['[ADDQ]'],tokenizer.vocab['[GPR64]'],3]
    assert tokenizer.encode('')==[2,3]
    print(f'PASS {len(cases)+2} tokenizer regression checks',flush=True)

if __name__=='__main__':
    check(json.loads((Path(__file__).resolve().parent/'tokenizer_spec.json').read_text()))
