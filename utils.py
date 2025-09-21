
import copy
from functools import lru_cache
from typing import Dict, List

from prefix_trie import CharacterPrefixTrie


@lru_cache(maxsize=2)
def create_prefix_trie(model_name):
    trie = CharacterPrefixTrie(model_name)
    trie.create_prefix_trie()
    return trie


@lru_cache
def _adapt_tokenizer(tokenizer):
    """Return a copy of the tokenizer with decoded vocab cached by id."""
    tokenizer = copy.deepcopy(tokenizer)

    decoded_vocab_by_id: Dict[int, str] = {}
    for token_id in tokenizer.get_vocab().values():
        decoded_vocab_by_id[token_id] = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    tokenizer.decoded_vocab_by_id = decoded_vocab_by_id
    tokenizer.decoded_vocab = [
        (token_str, token_id) for token_id, token_str in decoded_vocab_by_id.items()
    ]
    return tokenizer


def _get_decoded_vocab_by_id(tokenizer) -> Dict[int, str]:
    decoded_vocab_by_id = getattr(tokenizer, 'decoded_vocab_by_id', None)
    if decoded_vocab_by_id is None:
        raise RuntimeError(
            'Tokenizer must be adapted with _adapt_tokenizer before decoding token ids.'
        )
    return decoded_vocab_by_id
