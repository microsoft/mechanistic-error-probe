import torch
import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


def find_sub_list(sl,l, offset=0):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if ind < offset:
            continue
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def cosine(A, B):
    return np.dot(A,B.T)/(np.linalg.norm(A)*np.linalg.norm(B))

def find_within_text(prompt, parts, tokenizer):
    """
    A function that identifies the indices of tokens of a part of the prompt. 
    By default we use the first occurence. 
    """
    prompt_tokens = tokenizer.encode(prompt)
    part_tokens = [tokenizer.encode(p)[2:] for p in parts]
    part_token_indices = [find_sub_list(pt, prompt_tokens) for pt in part_tokens]
    return part_token_indices


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    From https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L225
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

