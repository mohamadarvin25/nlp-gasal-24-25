import math
from logging import lastResort

from nltk import trigrams
from zmq.backend import first

import preprocess_data
from itertools import chain


class NgramModel:
    def __init__(self, vocab: list[str], train: list[list[str]], test: list[list[str]]) -> None:
        self.vocab = vocab
        self.train = train
        self.test = test

    """
  - Fungsionalitas model ini adalah menghasilkan koleksi n-gram model.
  - Seperti 2-gram, artinya di dalam koleksi terdapat pasangan, seperti: '<s> saya', 'saya sedang', 'sedang makan', 'makan nasi', 'nasi kapau'.
  - Expected output berupa dictionary dengan pasangan key berupa n-length token serta value berupa kemunculan n-length token tersebut di dalam corpus.
  - Output format berupa dictionary.
  """

    def generate_n_grams(self, data: list[list[str]], n: int, start_token: str = '<s>', end_token='</s>') -> dict:
        # TODO: Implement based on the given description
        grams_count = {}
        for i,sentence in enumerate(data):
           sentence.insert(0,start_token)
           sentence.append(end_token)
           data[i] = sentence
        for sentence in data:
            first_index = 0
            last_index = n
            while last_index <= len(sentence):
                gram = tuple(sentence[first_index:last_index])

                grams_count[gram] = grams_count.get(gram, 0) + 1
                first_index += 1
                last_index += 1
        return grams_count

    """
  - Fungsionalitas method ini menghitung probabilitas suatu kata given kata/kumpulan kata.
  - Sederhananya, method ini merupakan implementasi dari ekspresi P(w_i|w_1:{i-1}).
  - Perlu diperhatikan bahwa pada parameter terdapat 'laplace_number' yang artinya Anda diharapkan mengimplementasikan add-one (laplace) smoothing.
  - Output format berupa float.
  """

    def count_probability(self, predicted_word: str, given_word: list[str], n_gram_counts, n_plus1_gram_counts,
                          vocabulary_size, laplace_number: float = 1.0) -> float:
        # TODO: Implement based on the given description
        example_key = next(iter(n_plus1_gram_counts))
        n_gram_type = len(example_key)
        count_predicted_word = 0
        count_given_word = 0
        if n_gram_type == 1:
            N = 0
            for key, value in n_plus1_gram_counts.items():
                N += value
                if given_word in key:
                    count_predicted_word = value
                    predicted_word_probability = (count_predicted_word + laplace_number)/(N+vocabulary_size)
                    return predicted_word_probability
        elif n_gram_type == 2:
            for key, value in n_gram_counts.items():
                if key[-1] == given_word[-1]:
                    count_given_word += n_gram_counts.get(key)
            for key, value in n_plus1_gram_counts.items():
                if key[-1] == predicted_word:
                    if key[-2] == given_word[-1]:
                        count_predicted_word += n_plus1_gram_counts.get(key)
            predicted_word_probability = (count_predicted_word + laplace_number) / (
                    count_given_word + vocabulary_size)
        else:
            for key, value in n_gram_counts.items():
                if key[-1] == given_word[-1]:
                    if key[-2] == given_word[-2]:
                        count_given_word += n_gram_counts.get(key)

            for key, value in n_plus1_gram_counts.items():
                if key[-1] == predicted_word:
                    if key[-2] == given_word[-1]:
                        if key[-3] == given_word[-2]:
                            count_predicted_word += n_plus1_gram_counts.get(key)
            predicted_word_probability = (count_predicted_word + laplace_number) / (
                    count_given_word + vocabulary_size)
        return predicted_word_probability


    """
  - Silakan Anda menggunakan method ini untuk bermain-main/menguji segala kemungkinan sentence/word generation berdasarkan method count_probability yang telah Anda bangun.
  """

    def probabilities_for_all_vocab(self, given_word: list[str], n_gram_counts, n_plus1_gram_counts, vocabulary,
                                    end_token='</s>', unknown_token='<unk>', laplace_number=1.0):
        example_key = next(iter(n_plus1_gram_counts))  # Get the first key from n_gram_counts
        n_gram_type = len(example_key)
        vocabulary = vocabulary + [end_token, unknown_token]
        vocab_size = len(vocabulary)
        probs = dict()
        if n_gram_type == 1:
            for word in given_word:
                prob = self.count_probability(word, word, n_gram_counts, n_plus1_gram_counts, vocab_size,
                                              laplace_number=laplace_number)
                probs[word] = prob
        else:
            for word in vocabulary:
                prob = self.count_probability(word, given_word, n_gram_counts, n_plus1_gram_counts, vocab_size,
                                              laplace_number=laplace_number)
                probs[word] = prob
        return probs

    """
  - Fungsionalitas pada method ini adalah untuk mengevaluasi n-gram model Anda menggunakan metrik perplexity.
  """

    def count_perplexity(self, sentence, n_gram_counts, n_plus1_gram_counts, vocab_size, vocab, start_token='<s>',
                         end_token='</s>', laplace_number=1.0):
        # TODO: Implement based on the given description
        sentence.insert(0, start_token)
        sentence.append(end_token)
        example_key = next(iter(n_plus1_gram_counts))
        n_gram_type = len(example_key)
        log_probability = 0.0
        probs = dict()
        if n_gram_type == 1:
            for word in sentence:
                prob = self.count_probability(word, word, n_gram_counts, n_plus1_gram_counts, vocab_size,
                                              laplace_number=laplace_number)
                probs[word] = prob
        else:
            for word in vocab:
                prob = self.count_probability(word, sentence, n_gram_counts, n_plus1_gram_counts, vocab_size,
                                              laplace_number=laplace_number)
                probs[word] = prob
        for key, value in probs.items():
            log_probability += math.log(value)
        perplexity = math.exp(-log_probability / len(sentence))
        return perplexity


def main():
    """
  EXAMPLE:
  - Pada contoh ini menggunakan scenario no lowercasing (cased)
  """
    lowercase: bool = False
    preprocess = preprocess_data.Preprocess()
    vocab, train, test = preprocess.load_from_pickle(lowercase)

    model = NgramModel(vocab, train, test)
    flatten_test = list(chain.from_iterable(test))

    """
  EXAMPLE:
  - Silakan berkreasi se-kreatif mungkin menggunakan beragam n-gram model yang Anda inginkan.
  - Anda dibebaskan untuk mengganti/menambah/menghapus contoh kombinasi di bawah ini sesuai dengan kreativitas Anda.
  - Kami sangat menghargai kreativitas Anda terkait Tugas Individu 2 ini.
  """
    unigram_counts = model.generate_n_grams(train, 1)
    bigram_counts = model.generate_n_grams(train, 2)
    trigram_counts = model.generate_n_grams(train, 3)
    """
  EXAMPLE:
  - Di bawah ini merupakan contoh/cara untuk generate kalimat dari n-gram LM
  """

    """
  # # UNIGRAM
  # # """
    def get_first_10_elements(dictionary):
        return dict(list(dictionary.items())[:100])
    # print(get_first_10_elements(unigram_counts))
    generate_S_unigram = model.probabilities_for_all_vocab(['rantai',"polinukleotida", "sampah"], train, get_first_10_elements(unigram_counts), vocab)
    print(max(generate_S_unigram, key=generate_S_unigram.get))
    print(sorted(generate_S_unigram.items(), key=lambda x: x[1], reverse=True)[:5])

    """
  BIGRAM
  """
    def get_first_10_elements(dictionary):
        return dict(list(dictionary.items())[:100])
    generate_S_bigram = model.probabilities_for_all_vocab(['polinukleotida'], get_first_10_elements(unigram_counts), get_first_10_elements(bigram_counts), vocab)
    print(max(generate_S_bigram, key=generate_S_bigram.get))
    print(sorted(generate_S_bigram.items(), key=lambda x: x[1], reverse=True)[:5])

    """
  TRIGRAM
  """

    print(get_first_10_elements(trigram_counts))
    generate_S_trigram = model.probabilities_for_all_vocab(["rantai","polinukleotida"],get_first_10_elements(bigram_counts), get_first_10_elements(trigram_counts), vocab)
    print(max(generate_S_trigram, key=generate_S_trigram.get))
    print(sorted(generate_S_trigram.items(), key=lambda x: x[1], reverse=True)[:5])

    """
  - Di bawah ini merupakan contoh/cara untuk menilai perplexity dari kalimat yang telah Anda generate dari langkah sebelumnya.
  """

    """
  UNIGRAM
  """
    perplexity_test_unigram = model.count_perplexity(flatten_test, train, unigram_counts, len(vocab), vocab,
                                                     laplace_number=1.0)
    print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

    """
  # BIGRAM
  # """
    perplexity_test_bigram = model.count_perplexity(flatten_test, unigram_counts, bigram_counts, len(vocab), vocab,
                                                    laplace_number=1.0)
    print(f"n = 1, Perplexity: {perplexity_test_bigram:.4f}")

    """
  # TRIGRAM
  # """
    perplexity_test_trigram = model.count_perplexity(flatten_test, bigram_counts, trigram_counts, len(vocab), vocab,
                                                     laplace_number=1.0)
    print(f"n = 2, Perplexity: {perplexity_test_trigram:.4f}")

    """
  # TRIGRAM from generated sentence
  # """
    perplexity_test_random = model.count_perplexity(
        ['<s>', 'cagar', 'budaya', 'merupakan', 'aset', 'di', 'indonesia', '</s>'], bigram_counts, trigram_counts, vocab,
        len(vocab), laplace_number=1.0)
    print(f"n = 2, Perplexity: {perplexity_test_random:.4f}")


if __name__ == "__main__":
    main()
