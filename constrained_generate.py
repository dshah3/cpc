from vllm import LLM
from vllm.sampling_params import SamplingParams, LogitsProcessor
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

class SimpleLogitsProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        mask = torch.full_like(logits, float('-inf'))
        for token_id in range(len(logits)):
            token_str = self.tokenizer.decode([token_id])
            if 'a' in token_str.lower():
                mask[token_id] = 0

        masked_logits = mask + logits

        top5_before = torch.topk(logits, 5)
        top5_after = torch.topk(masked_logits, 5)
        print("Top 5 tokens before masking:")
        for i, (score, idx) in enumerate(zip(top5_before.values, top5_before.indices)):
            token = self.tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (score: {score:.2f})")
        
        print("Top 5 tokens after masking:")
        for i, (score, idx) in enumerate(zip(top5_after.values, top5_after.indices)):
            token = self.tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (score: {score:.2f})")
        
        print("-" * 50)
        
        return masked_logits

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    sampling_params = SamplingParams(max_tokens=3)

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
    )

    processor = SimpleLogitsProcessor(AutoTokenizer.from_pretrained(model_name))

    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=1.0,
        logits_processors=[processor],
    )

    prompt = "The weather today is"
    print(f"Prompt: '{prompt}'")
    print("=" * 50)
    print("Generating with constraint: only tokens containing 'a' allowed")
    print("=" * 50)
    
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    
    print("\nFinal output:")
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
