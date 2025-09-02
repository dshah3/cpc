from transformers import AutoTokenizer

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.token_ids = set()

class CharacterPrefixTrie:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab = self.tokenizer.vocab
        self.root = TrieNode()

    def create_prefix_trie(self):
        for token_str, token_id in self.vocab.items():
            node = self.root
            for char in token_str:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.token_ids.add(token_id)
    
    def find_tokens_that_are_prefixes_of(self, text: str):
        allowed_token_ids = set()
        node = self.root

        for char in text:
            if char not in node.children:
                break

            node = node.children[char]

            if node.is_word:
                allowed_token_ids.update(node.token_ids)

        return allowed_token_ids
    
    def find_all_tokens_starting_with(self, prefix: str) -> set:
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return set()
            
        allowed_ids = set()
        self._collect_all_tokens_from_node(node, allowed_ids)
        return allowed_ids
    
    def _collect_all_tokens_from_node(self, node: TrieNode, collection_set: set):
        if node.is_word:
            collection_set.update(node.token_ids)

        for child_node in node.children.values():
            self._collect_all_tokens_from_node(child_node, collection_set)
