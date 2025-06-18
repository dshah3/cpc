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

if __name__ == "__main__":
    trie = CharacterPrefixTrie("mistralai/Mistral-7B-Instruct-v0.3")

    trie.create_prefix_trie()

    prefix = "\u2581want"
    node = trie.get_node_for_prefix(prefix)
    if node:
        trie.print_trie(node=node, prefix=prefix)
    else:
        print(f"No tokens found with prefix '{prefix}'")

