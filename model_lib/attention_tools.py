import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from .hooks import TraceDict
from .misc_utils import repeat_kv, find_within_text

@torch.no_grad()
def run_attention_monitor(prompt_info,
                          model_wrapped,
                          max_new_tokens=15):
    """
    Args:
        prompt_info: should have a "prompt" and "constraints" field.
        model_wrapped : A model object using our wrapper: model_lib.hf_tooling.HF_LM
        max_new_tokens (int, optional): Maximum number of tokens to sample. Defaults to 15.

    Returns:
        A list of dictionaries, each containing the attention weights, hidden states, and other relevant information.
    """
    o_proj_matrices = model_wrapped.get_output_projection_matrices()

    model_config = model_wrapped.model.config
    num_key_value_groups = model_config.num_attention_heads // model_config.num_key_value_heads
    attention_layers = [(i, f"model.layers.{i}.self_attn") for i in range(model_wrapped.model.config.num_hidden_layers)]
    # mlp_layers = [(i, f"model.layers.{i}.mlp") for i in range(model_wrapped.model.config.num_hidden_layers)]
    all_layers = attention_layers # + mlp_layers
    
    prompt = prompt_info["prompt"]
    with torch.no_grad():
        # Sample the model's completion
        completion = model_wrapped.sample([prompt], max_new_tokens=max_new_tokens)[0]
                
    with torch.no_grad(), TraceDict(model_wrapped.model, [l[1] for l in all_layers]) as ret:
        # Run the full text through the model.
        inputs = model_wrapped.tokenizer(prompt+completion, return_tensors="pt").to(model_wrapped.device)
        outs = model_wrapped.model(input_ids=inputs.input_ids, 
                                          attention_mask=inputs.attention_mask,
                               output_hidden_states=True,
                                  output_attentions=True)  
        num_prompt_tokens = len(model_wrapped.tokenizer.encode(prompt))

        att_weights, proj_contributions, token_contribs = [], [], []
        for (l, layername) in all_layers:
            if "self_attn" in layername:
                o_proj = o_proj_matrices[l]
                rep_layer = ret[layername].output
                # A^{i,j}_{l} will be H x T x T
                att_weights.append(rep_layer[1].cpu().float().numpy()[0])
                # past key value states
                pkv = rep_layer[2][1][0]
                # compute |A_{i, j}*h_{j}*W_{v}*W_{o}|
                # Which will be H x T x D
                pkv = repeat_kv(pkv, num_key_value_groups)[0]
                proj_contributions.append(torch.einsum("HDd, HTd->HTD", o_proj, pkv).detach().cpu().float().numpy())
                token_contribs.append((att_weights[-1][:, num_prompt_tokens-1, :, np.newaxis]*proj_contributions[-1][:, :, :]).sum(axis=0))

            #elif "mlp" in layername:
            #   We played around with MLP contributions as well, but it's not included in the final version.
            #    mlp_contribs = ret[layername].output
            #    mlps.append(mlp_contribs.cpu().float().numpy()[0])
    
    data = edict({
            "completion": completion,
            "full_prompt": prompt+completion,
            "num_prompt_tokens": num_prompt_tokens,
            "prompt_tokens": model_wrapped.tokenizer.encode(prompt),
            **prompt_info
        })
    att_weights = np.stack(att_weights)
    proj_contribs = np.stack(proj_contributions)
    token_contribs = np.stack(token_contribs)
    
    constraint_indices = find_within_text(data.prompt, data.constraints, model_wrapped.tokenizer)
    # Get the locations of the filler tokens and template tokens.  
    generation_start = num_prompt_tokens-1
    generation_end = att_weights.shape[-1]-1
    
    data.token_labels = [model_wrapped.tokenizer.decode(t) for t in data.prompt_tokens]
    data.all_max_attention_weights = np.max(att_weights[:, :, generation_start], axis=1)
    data.all_token_contrib_norms = np.linalg.norm(token_contribs, axis=-1)
    
    data.token_contrib_norms_constraints = []
    data.attention_weights_constraints = []
    for (constraint_start, constraint_end) in constraint_indices:
        attention_weights_constraints = att_weights[:, :, generation_start, constraint_start:constraint_end+1]

        token_contribs_constraints = (attention_weights_constraints[:, :, :, np.newaxis]*proj_contribs[:, :, constraint_start:constraint_end+1])
        token_contrib_norms_constraints = np.linalg.norm(token_contribs_constraints, axis=-1)
        
        data.token_contrib_norms_constraints.append(token_contrib_norms_constraints)
        data.attention_weights_constraints.append(attention_weights_constraints)
    
    return data