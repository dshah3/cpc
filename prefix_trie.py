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

    def print_trie(self, node=None, prefix=""):
        if node is None:
            node = self.root

        if node.is_word:
            print(f"Token string: '{prefix}', Token IDs: {node.token_ids}")

        for char, child in node.children.items():
            self.print_trie(child, prefix + char)

    def get_node_for_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
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

if __name__ == "__main__":
    trie = CharacterPrefixTrie("mistralai/Mistral-7B-Instruct-v0.3")

    trie.create_prefix_trie()

    prefix = "▁I ▁want ▁to ▁live ▁acros"
    for i in range(len(prefix), -1, -1):
        print(prefix[i:])
        allowed_tokens = trie.find_all_tokens_starting_with(prefix[i:])
        print(f"Prefix: {prefix[i:]}, Number of allowed tokens: {len(allowed_tokens)}")

