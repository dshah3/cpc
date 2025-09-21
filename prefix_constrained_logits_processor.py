from prefix_trie import CharacterPrefixTrie
from typing import List, Tuple, Set
import torch

class PrefixConstrainedLogitsProcessor:

    def __init__(self, tokenizer, trie: CharacterPrefixTrie, remaining_prefix: str):
        self.tokenizer = tokenizer
        self.trie = trie
        self.remaining_prefix = remaining_prefix
        self.prefix_consumed = False

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        if self.prefix_consumed:
            return logits

        if not self.remaining_prefix:
            self.prefix_consumed = True
            return logits

        allowed_token_ids = self._allowed_token_ids_for_remaining_prefix()
        if not allowed_token_ids:
            return torch.full_like(logits, float('-inf'))

        return self._apply_mask(logits, allowed_token_ids)

    def update_remaining_prefix(self, generated_token_id: int):
        if self.prefix_consumed:
            return

        token_str = get_decoded_token_str(self.tokenizer, generated_token_id)

        if self.remaining_prefix.startswith(token_str):
            self.remaining_prefix = self.remaining_prefix[len(token_str):]

            if not self.remaining_prefix:
                self.prefix_consumed = True

        elif token_str.startswith(self.remaining_prefix):
            self.remaining_prefix = ""
            self.prefix_consumed = True

    def _allowed_token_ids_for_remaining_prefix(self) -> Set[int]:
        allowed: Set[int] = set()
        allowed.update(self.trie.find_tokens_that_are_prefixes_of(self.remaining_prefix))
        allowed.update(self.trie.find_all_tokens_starting_with(self.remaining_prefix))
        return allowed

    def _apply_mask(self, logits: torch.Tensor, allowed_token_ids: Set[int]) -> torch.Tensor:
        mask = torch.full_like(logits, float('-inf'))
        indices = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=logits.device)
        mask.index_fill_(0, indices, 0)
        return logits + mask