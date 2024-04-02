import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class HF_Llama2_Wrapper:
    def __init__(self, model, tokenizer, device="cuda", load_in_8bit=False):
        """A wrapper around HuggingFace language models to compute likelihoods and generate text.
        Args:
            model: a Llama model from HuggingFace.
            tokenizer : a tokenizer from HuggingFace.
            device (str): Defaults to "cuda". Not extensively tested for CPUs.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        # Careful: Below are tested for Llama-2 family.
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.load_in_8bit = load_in_8bit

    
    @torch.no_grad()
    def get_batch_loglikelihood(self, texts):
        """
        Compute the loglikelihood of the given set of texts.
        """
        perplexities = []
        total_likelihoods = []
        for text in texts:
            tokenized = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
            labels = tokenized.input_ids
            outputs = self.model(input_ids=tokenized["input_ids"], 
                                 attention_mask=tokenized["attention_mask"], 
                                 labels=labels)
            logits = outputs.logits.cpu()
            labels = labels.cpu()

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
            ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
            nonpad_per_row = (shift_labels != -100).sum(dim=1)
            ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row

            ll_per_sample[(shift_labels == -100)] = 0
            ll_total = ll_per_sample.sum(dim=1)
            perplexities.append(ll_mean.cpu().numpy())
            total_likelihoods.append(ll_total.cpu().numpy())
        perplexities = np.concatenate(perplexities, axis=0)
        total_likelihoods = np.concatenate(total_likelihoods, axis=0)
        return perplexities, total_likelihoods 
    
    
    
    @torch.no_grad()
    def get_conditional_loglikelihood(self, texts: list, 
                                      completion_offset: torch.Tensor):
        tokenized = self.tokenizer(
            texts, return_tensors="pt", truncation=True, return_token_type_ids=False, padding=True,
        ).to(self.device)
        labels = tokenized.input_ids
        outputs = self.model(**tokenized)
        logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
        labels = labels.cpu()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        
        # Ignore the tokens prior to the completion offset. 
        mask = (torch.arange(shift_labels.size(1)).unsqueeze(0) < (shift_labels.shape[1]-completion_offset.unsqueeze(1)))
        # Choose which tokens to mask from LL computation.
        shift_labels[(mask | (shift_labels == self.tokenizer.pad_token_id))] = -100
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
        ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
        #nonpad_per_row = (shift_labels != -100).sum(dim=1)
        #ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row
        
        ll_per_sample[(shift_labels == -100)] = 0
        ll_total = ll_per_sample.sum(dim=1)
        torch.cuda.empty_cache()
        return ll_per_sample.float().cpu().numpy(), ll_total.float().cpu().numpy()
    
    @torch.no_grad()
    def get_conditional_loglikelihood_batch(self, texts: list,
                                            completion_offset: torch.Tensor):
        """Padding messes with the likelihood computation. Will do 1 by 1 for simplicity, but this is not a great way."""
        ll_mean_all = []
        ll_total_all = []
        for i in tqdm(range(len(texts))):
            llm, llt = self.get_conditional_loglikelihood(texts[i:i+1], completion_offset[i:i+1])
            ll_mean_all.append(llm)
            ll_total_all.append(llt)
        return ll_mean_all, np.concatenate(ll_total_all, axis=0)
    
    @torch.no_grad()
    def generate(self, texts, *args, **kwargs):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        generated = self.model.generate(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], 
                                        pad_token_id=self.tokenizer.pad_token_id, *args, **kwargs)
        torch.cuda.empty_cache()
        return generated

    @torch.no_grad()
    def _sample_api_single(self, text, *args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs["max_tokens"]
            kwargs.pop("max_tokens")
        generated = self.generate([text], do_sample=False, temperature=None,
                     output_scores=True, return_dict_in_generate=True, top_p=None, *args, **kwargs)
        decoded = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        target_completion = decoded.replace(text, "")
        return target_completion
    
    @torch.no_grad()
    def sample(self, texts, *args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs["max_tokens"]
            kwargs.pop("max_tokens")
        generated = self.generate(texts, do_sample=False, num_beams=1,
                     output_scores=True, return_dict_in_generate=True, temperature=None, top_p=None, *args, **kwargs)
        decoded = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        target_completions = [d.replace(t, "") for d, t in zip(decoded, texts)]
        return target_completions       
    
    def get_output_projection_matrices(self):
        """Get the output projection matrices for the model."""
        
        if self.load_in_8bit:
            # If the model is loaded in 8bit, we need to dequantize the output projection matrices for the downstream usage.
            o_proj_matrices = []
            for nm, p in self.model.named_modules():
                if "o_proj" not in nm: 
                    continue
                weight = p.weight
                dequantized = (weight.CB * weight.SCB) / 127
                o_proj_matrices.append(torch.stack(dequantized.split(self.model.config.hidden_size // self.model.config.num_attention_heads, dim=1)).detach().bfloat16())

        else:
            o_proj_matrices = [torch.stack(p.weight.split(self.model.config.hidden_size // self.model.config.num_attention_heads, dim=1)).detach() for nm, p in self.model.named_modules() if "o_proj" in nm]
        return o_proj_matrices