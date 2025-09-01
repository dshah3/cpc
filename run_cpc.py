from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from typing import List, Tuple, Optional
from functools import lru_cache

from prefix_trie import CharacterPrefixTrie

class PrefixConstrainedLogitsProcessor:

    def __init__(self, tokenizer, trie: CharacterPrefixTrie, remaining_prefix: str):
        self.tokenizer = tokenizer
        self.trie = trie
        self.remaining_prefix = remaining_prefix
        self.prefix_consumed = False

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        if self.prefix_consumed:
            return logits
            
        # Create mask with -inf for all tokens
        mask = torch.full_like(logits, float('-inf'))
        
        if self.remaining_prefix:
            # Find allowed tokens
            allowed_token_ids = self.trie.find_all_tokens_starting_with(self.remaining_prefix)
            prefix_token_ids = self.trie.find_tokens_that_are_prefixes_of(self.remaining_prefix)
            allowed_token_ids.update(prefix_token_ids)
            
            # Debug info
            print(f"\nRemaining prefix: '{self.remaining_prefix}'")
            print(f"Allowed tokens: {len(allowed_token_ids)}")
            
            return masked_logits
                
        else:
            self.prefix_consumed = True
            return logits

    def update_remaining_prefix(self, generated_token_id: int):
        if self.prefix_consumed:
            return

        token_str = self.tokenizer.decode([generated_token_id])

        if self.remaining_prefix.startswith(token_str):
            self.remaining_prefix = self.remaining_prefix[len(token_str):]
            print(f"Token '{token_str}' matched prefix. Remaining: '{self.remaining_prefix}'")

            if not self.remaining_prefix:
                self.prefix_consumed = True

        elif token_str.startswith(self.remaining_prefix):
            self.remaining_prefix = ""
            self.prefix_consumed = True
                
        else:
            print(f"WARNING: Token '{token_str}' doesn't match prefix '{self.remaining_prefix}'")


def find_tokenization_breakpoint(tokenizer, trie: CharacterPrefixTrie, prefix: str) -> Tuple[List[int], str]:
    last_valid_position = 0
    for i in range(len(prefix), -1, -1):
        remaining = prefix[i:]
        if remaining:
            allowed_tokens = trie.find_all_tokens_starting_with(remaining)
            if len(allowed_tokens) > 0:
                last_valid_position = i

    clean_prefix = prefix[:last_valid_position]
    remaining_prefix = prefix[last_valid_position:]

    if clean_prefix:
        clean_token_ids = tokenizer.encode(clean_prefix, add_special_tokens=False)
    else:
        clean_token_ids = []

    return clean_token_ids, remaining_prefix

@lru_cache(maxsize=2)
def create_prefix_trie(model_name):
    trie = CharacterPrefixTrie(model_name)
    trie.create_prefix_trie()
    return trie


def generate_with_prefix_constraint(
    model_name: str,
    prompt: str,
    character_prefix: str,
    max_new_tokens: int = 50
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trie = create_prefix_trie(model_name)

    prefix_tokens, remaining_prefix = find_tokenization_breakpoint(tokenizer, trie, character_prefix)

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
    )

    full_prompt = prompt + tokenizer.decode(prefix_tokens)

    processor = PrefixConstrainedLogitsProcessor(tokenizer, trie, remaining_prefix)

    class UpdateCallback:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, token_id):
            self.processor.update_remaining_prefix(token_id)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        logits_processors=[processor],
    )

    outputs = llm.generate([full_prompt], sampling_params=sampling_params)

    generated_text = outputs[0].outputs[0].text
    generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)

    for i, token_id in enumerate(generated_tokens[:len(remaining_prefix)+2]):
        processor.update_remaining_prefix(token_id)

    full_output = full_prompt + generated_text

    return full_output


def generate_with_prefix_constraint_simple(
    model_name: str,
    prompt: str,
    character_prefix: str,
    max_new_tokens: int = 150,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trie = create_prefix_trie(model_name)

    prefix_tokens, remaining_prefix = find_tokenization_breakpoint(tokenizer, trie, character_prefix)

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
    )

    current_text = prompt + tokenizer.decode(prefix_tokens)
    remaining = remaining_prefix

    tokens_generated = 0

    while remaining and tokens_generated < max_new_tokens:
        processor = PrefixConstrainedLogitsProcessor(tokenizer, trie, remaining)
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            logits_processors=[processor],
        )

        outputs = llm.generate([current_text], sampling_params=sampling_params)
        new_token_text = outputs[0].outputs[0].text

        new_tokens = tokenizer.encode(new_token_text, add_special_tokens=False)
        if new_tokens:
            new_token_id = new_tokens[0]
            
            token_str = tokenizer.decode([new_token_id])
            
            if remaining.startswith(token_str):
                remaining = remaining[len(token_str):]
            elif token_str.startswith(remaining):
                remaining = ""
            
            current_text += new_token_text
            tokens_generated += 1

    if tokens_generated < max_new_tokens:
        remaining_tokens = max_new_tokens - tokens_generated
        
        print(f"\nPrefix satisfied! Generating {remaining_tokens} more tokens normally...")
        
        sampling_params = SamplingParams(
            max_tokens=remaining_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        
        outputs = llm.generate([current_text], sampling_params=sampling_params)
        current_text += outputs[0].outputs[0].text

    return current_text


def analyze_prefix_breakpoints(model_name: str, test_prefixes: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trie = create_prefix_trie(model_name)

    for prefix in test_prefixes:
        clean_tokens, remaining = find_tokenization_breakpoint(tokenizer, trie, prefix)
        print(f"\nSummary:")
        print(f"  Original: '{prefix}'")
        print(f"  Clean part: '{tokenizer.decode(clean_tokens)}'")
        print(f"  Remaining: '{remaining}'")
        print(f"  Clean tokens: {clean_tokens}")
        
        # Show what tokens can start with the remaining prefix
        if remaining:
            allowed = trie.find_all_tokens_starting_with(remaining)
            print(f"  Tokens starting with '{remaining}': {len(allowed)}")
            if allowed and len(allowed) < 10:
                for tid in list(allowed)[:5]:
                    print(f"    - '{tokenizer.decode([tid])}'")


def main():
    test_prefixes = [
        "import numpy as n",
        "The weather is beau",
        "def fibonacci",
        "Hello, wor",
        "import tensorflow.keras.lay"
    ]

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    analyze_prefix_breakpoints(model_name, test_prefixes)

    test_cases = [
        # {
        #     "prompt": "# Python code to work with arrays\n",
        #     "prefix": "import numpy as n",
        #     "description": "Autocomplete numpy import and rest of function"
        # },
        {
            "prompt": "Please describe the weather in Tunisia back 30 years ago. The weather was",
            "prefix": "absolutel",
            "description": "Complete 'absolutely beau' and tell us about weather patterns in Tunisia."
        },
        {
            "prompt": "Write a layernorm function in Keras. <code>",
            "prefix": "import tensorflow.keras.lay",
            "description": "Complete function definition"
        }
    ]

    for test in test_cases:
        full_output = generate_with_prefix_constraint_simple(
            model_name=model_name,
            prompt=test["prompt"],
            character_prefix=test["prefix"],
            max_new_tokens=100,
        )
        print(f"FULL OUTPUT: {full_output}")





if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()