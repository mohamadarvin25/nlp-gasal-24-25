import time
import sys

from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

# Code is modified from http://stevehanov.ca/blog/?id=114 and paper 10.1109/ISRITI56927.2022.10053062


class DamerauLevenshteinDict:
    def __init__(self, dict_path):
        self._kbbi = self.convert_dict(dict_path)

    def convert_dict(self, path):
        # convert KBBI into Dict structure --> {'word':1}, 1 just dummy number
        kbbi_dict = {line.split()[0]: 1 for line in open(
            path, encoding='utf-8')}
        return kbbi_dict

    def search(self, typo, max_cost):
        results = []
        for candidate_word in self._kbbi.keys():
            cost = damerau_levenshtein_distance(typo, candidate_word)

            if cost <= max_cost:
                results.append((candidate_word, cost))
        return results

    def get_candidates(self, typo, max_cost):
        candidates = self.search(typo, max_cost)
        sorted_candidates = sorted(
            candidates, key=lambda item: (item[1], len(item), item[0]))
        candidate_words = [candidate[0] for candidate in sorted_candidates]

        return candidate_words
