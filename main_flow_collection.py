import os
import argparse
import torch
import pickle
import transformers 

from tqdm import tqdm
from easydict import EasyDict as edict

from model_lib.hf_tooling import HF_Llama2_Wrapper
from model_lib.attention_tools import run_attention_monitor
from factual_queries import load_constraint_dataset


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_new_tokens", type=int, default=15, help="Number of tokens to generate for each prompt.")
    parser.add_argument("--dataset-name", type=str, default="basketball_players")
    parser.add_argument("--load-in-8bit", action="store_true", help="Whether to load the model in 8-bit mode. We used this only for Llama-2 70B.")
    parser.add_argument("--subsample-count", type=int, default=None, help="Number of items to run for, mostly for testing mode.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory to save the attention flow.")
    return parser.parse_args()

args = config()

## Load the model and tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name,
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16,
                                            load_in_8bit=args.load_in_8bit,
                                            device_map="auto")
model_wrapped = HF_Llama2_Wrapper(model, tokenizer, device="cuda")

items = load_constraint_dataset(args.dataset_name, subsample_count=args.subsample_count)

print(f"Will run for {len(items)} items")

records_to_save = edict({"token_contrib_norms_constraints": [],  "attention_weights_constraints": [],
                         "gt_logprob": [], "pred_logprob": [], "constraint": [], "prompt": [], "popularity": [],
                         })

for item in tqdm(items):
    prompt_info = {"prompt": item["prompt"]}
    if "constraints" in item:
        prompt_info["constraints"] = item["constraints"]
    else:
        prompt_info["constraints"] = [f" {item['constraint']}"]

    data = run_attention_monitor(prompt_info,
                                 model_wrapped, args.max_new_tokens)
    
    print(item, data["completion"])
    # Completing the likelihoods of the completion vs the ground truth
    if "label" in item:
        # Multi-constraint data do not have pre-defined labels. We run different verifiers.
        ground_truth = " " + str(item["label"])
        completion_len_gt = len(model_wrapped.tokenizer(ground_truth)["input_ids"][1:]) # Offset by 1 to account for <s>
        completion_tokenized = model_wrapped.tokenizer(data["completion"])
        completion_cut = tokenizer.decode(completion_tokenized["input_ids"][1:1+completion_len_gt])
        
        prompts_logprob = [item["prompt"] + ground_truth, item["prompt"] + completion_cut]
        completion_offset = torch.tensor([completion_len_gt, completion_len_gt])
        _, loglikelihoods = model_wrapped.get_conditional_loglikelihood_batch(texts=prompts_logprob,
                                                                            completion_offset=completion_offset)
        records_to_save.gt_logprob.append(loglikelihoods[0])
        records_to_save.pred_logprob.append(loglikelihoods[1])        
    
    # Saving these records for later probing analysis.
    records_to_save.attention_weights_constraints.append(data.attention_weights_constraints)
    records_to_save.token_contrib_norms_constraints.append(data.token_contrib_norms_constraints)
    records_to_save.popularity.append(item["popularity"])

    records_to_save.prompt.append(item["prompt"])
    records_to_save.constraint.append(prompt_info["constraints"])

os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_{args.dataset_name}.pkl")
with open(output_file, "wb") as f:
    pickle.dump(records_to_save, f)
