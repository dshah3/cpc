from utils import _adapt_tokenizer, create_prefix_trie
from prefix_constrained_logits_processor import PrefixConstrainedLogitsProcessor

from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from typing import List, Tuple
from prefix_trie import CharacterPrefixTrie

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


def generate_with_prefix_constraint(
    model_name: str,
    prompt: str,
    character_prefix: str,
    max_new_tokens: int = 150,
) -> str:
    tokenizer = _adapt_tokenizer(AutoTokenizer.from_pretrained(model_name))
    trie = create_prefix_trie(model_name)

    prefix_tokens, remaining_prefix = find_tokenization_breakpoint(tokenizer, trie, character_prefix)

    current_text = prompt + tokenizer.decode(
        prefix_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    ) if prefix_tokens else prompt

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
    )

    tokens_generated = 0

    while remaining_prefix and tokens_generated < max_new_tokens:
        processor = PrefixConstrainedLogitsProcessor(tokenizer, trie, remaining_prefix)
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=5,
            logits_processors=[processor],
        )

        outputs = llm.generate([current_text], sampling_params=sampling_params)
        new_text = outputs[0].outputs[0].text
        current_text += new_text
        tokens_generated += 1

        token_ids = tokenizer.encode(new_text, add_special_tokens=False)
        if not token_ids:
            break

        processor.update_remaining_prefix(token_ids[0])
        remaining_prefix = processor.remaining_prefix

    tokens_left = max_new_tokens - tokens_generated
    if tokens_left > 0 and not remaining_prefix:
        sampling_params = SamplingParams(
            max_tokens=tokens_left,
            temperature=0.0,
            top_p=1.0,
        )

        outputs = llm.generate([current_text], sampling_params=sampling_params)
        current_text += outputs[0].outputs[0].text

    return current_text


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    test_cases = [
        {
            "prompt": "Write a layernorm function in Keras. <code>",
            "prefix": "import tensorflow.keras.lay",
            "description": "Complete function definition",
        },
    ]

    for test in test_cases:
        full_output = generate_with_prefix_constraint(
            model_name=model_name,
            prompt=test["prompt"],
            character_prefix=test["prefix"],
            max_new_tokens=100,
        )
        print(f"FULL OUTPUT: {full_output}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
