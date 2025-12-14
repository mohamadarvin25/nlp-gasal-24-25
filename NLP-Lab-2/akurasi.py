def hitung_akurasi(tokenizer_tokens_list: list[list[str]], gold_std_tokens_list: list[list[str]]) -> float:
    """
    Fungsi ini menghitung akurasi dari hasil tokenisasi suatu tokenizer
    terhadap hasil tokenisasi gold standard (asumsikan bahwa fungsi ini
    menerima hasil tokenisasi dari semua teks di dataset testing sekaligus)
    """
    # Total token yang benar
    total_true_token = 0

    # Total token di gold standard
    total_gold_token = 0

    # Iterasi tiap kalimat
    for tokenizer_tokens, gold_tokens in zip(tokenizer_tokens_list, gold_std_tokens_list):
        # String untuk konkatinasi tiap token di tokenizer
        string_tokenizer = ""

        # String untuk konkatinasi tiap token di gold token
        string_gold = ""

        # index token pada suatu kalimat di tokenizer
        tokenizer_index = 0

        # index token pada suatu kalimat di gold token
        gold_index = 0

        # Ketika semua token belum di dibandingkan
        while tokenizer_index < len(tokenizer_tokens) and gold_index < len(gold_tokens):
            tokenizer_token = tokenizer_tokens[tokenizer_index]
            gold_token = gold_tokens[gold_index]

            if tokenizer_token == '[UNK]':
                tokenizer_token = '_'

            if string_tokenizer == string_gold:
                if tokenizer_token == gold_token:
                    total_true_token += 1 # Token benar, tambahkan

                total_gold_token += 1 # Total token di gold bertambah
                tokenizer_index += 1  # Maju ke token berikutnya
                gold_index += 1       # Maju ke token berikutnya

                # Penambahan token diikuti penambahan string
                string_tokenizer += tokenizer_token
                string_gold += gold_token

            # Jika string tokenizer lebih pendek
            elif string_tokenizer < string_gold:
                # Penambahan string dan index tokenizer
                string_tokenizer += tokenizer_token
                tokenizer_index += 1

            # Jika string gold lebih pendek
            else:
                string_gold += gold_token
                gold_index += 1
                total_gold_token += 1

    return round((total_true_token / total_gold_token), 2)


if __name__ == "__main__":
    # Contoh pemanggilan akurasi seperti pada dokumen soal
    tokenizer_tokens_list = [
        ["Bukunya", "mahal", "."],
        ["Seharus", "nya", "kamu", "tidak", "terlambat", "."],
    ]

    gold_std_tokens_list = [
        ["Buku", "nya", "mahal", "."],
        ["Seharusnya", "kamu", "tidak", "terlambat", "."],
    ]

    epsilon = 1e-7
    assert abs(hitung_akurasi(tokenizer_tokens_list,
               gold_std_tokens_list) - 0.67) <= epsilon
