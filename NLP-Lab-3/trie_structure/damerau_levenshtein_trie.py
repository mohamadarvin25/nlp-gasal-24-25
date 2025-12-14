from trie_structure.trie import TrieNode

# Code is modified from http://stevehanov.ca/blog/?id=114 and paper 10.1109/ISRITI56927.2022.10053062


class DamerauLevenshteinTrie:
    def __init__(self, dict_path):
        self.trie = TrieNode()
        self._dict = self.convert_trie(dict_path)

    def convert_trie(self, path):
        for word in open(path, "rt", encoding='utf-8').read().split():
            self.trie.insert(word)

    def search(self, word, max_cost):
        word = word.lower()
        current_row = range(len(word)+1)
        results = []

        for letter in self.trie.children:
            self.search_recursive(
                self.trie.children[letter], letter, None, word, current_row, None, results, max_cost)

        return results

    def search_recursive(
        self,
        node,
        char,
        prev_char,
        word,
        previous_row,
        pre_previous_row,
        results,
        max_cost,
    ):
        columns = len(word) + 1
        current_row = [previous_row[0] + 1]

        for column in range(1, columns):
            delete_cost = previous_row[column] + 1
            insert_cost = current_row[column - 1] + 1

            if word[column - 1] != char:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row.append(min(insert_cost, delete_cost, replace_cost))
            if (
                prev_char
                and column - 1 > 0
                and char == word[column - 2]
                and prev_char == word[column - 1]
                and word[column - 1] != char
            ):
                current_row[column] = min(
                    current_row[column], pre_previous_row[column - 2] + 1
                )

        if current_row[-1] <= max_cost and node.word != None:
            results.append((node.word, current_row[-1]))

        if min(current_row) <= max_cost:
            prev_char = char
            for char in node.children:
                self.search_recursive(
                    node.children[char],
                    char,
                    prev_char,
                    word,
                    current_row,
                    previous_row,
                    results,
                    max_cost,
                )

    def get_candidates(self, typo, max_cost):

        candidates = self.search(typo, max_cost)
        sorted_candidates = sorted(
            candidates, key=lambda item: (item[1], len(item), item[0]))

        candidate_words = [candidate[0] for candidate in sorted_candidates]

        return candidate_words
