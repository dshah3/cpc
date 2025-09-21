from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from typing import List, Tuple, Set
from functools import lru_cache
import copy

from prefix_trie import CharacterPrefixTrie


@lru_cache(maxsize=2)
def create_prefix_trie(model_name):
    trie = CharacterPrefixTrie(model_name)
    trie.create_prefix_trie()
    return trie

@lru_cache
def _adapt_tokenizer(tokenizer):
    """
    Adapt tokenizer by caching the decoded vocabulary
    """
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.decoded_vocab = [(tokenizer.decode(tok_id), tok_id) for _, tok_id in tokenizer.get_vocab().items()]
    return tokenizer


def get_decoded_token_str(tokenizer, token_id: int) -> str:
    decoded_vocab = getattr(tokenizer, "decoded_vocab", None)
    if decoded_vocab is not None:
        if not hasattr(tokenizer, "decoded_vocab_by_id"):
            tokenizer.decoded_vocab_by_id = {
                tok_id: tok_str for tok_str, tok_id in decoded_vocab
            }
        token_str = tokenizer.decoded_vocab_by_id.get(token_id)
        if token_str is not None:
            return token_str

    return tokenizer.decode([token_id])


def token_ids_to_text(tokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return ""

    if len(token_ids) == 1:
        return get_decoded_token_str(tokenizer, token_ids[0])

    return tokenizer.decode(token_ids)

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

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
    )

    current_text = prompt + token_ids_to_text(tokenizer, prefix_tokens)
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
        new_text = outputs[0].outputs[0].text
        current_text += new_text
        tokens_generated += 1

        token_ids = tokenizer.encode(new_text, add_special_tokens=False)
        if not token_ids:
            break

        processor.update_remaining_prefix(token_ids[0])
        remaining = processor.remaining_prefix

    tokens_left = max_new_tokens - tokens_generated
    if tokens_left > 0 and not remaining:
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
