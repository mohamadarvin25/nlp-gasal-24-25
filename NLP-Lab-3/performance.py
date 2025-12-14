import time


class Performance:
    """
    Menghitung peforma tiap model dengan menggunakan metriks Best match accuracy dan
    Candidate accuracy

    atribut:
    model = Jenis algoritma yang digunakan untuk pencarian kandidat kata yang benar
    typo_sample_dataset = list 2 dimensi yang tiap row nya mengandung typo sample
    keys dan list kemungkinan non words error nya
    """
    def __init__(self, model, typo_sample_dataset):
        self.__model = model
        self.__typo_sample_dataset = typo_sample_dataset

    def calculate_performance(self):
        """
        Memanggil semua jenis fungsi perhitungan akurasinya dan menghitung runtimenya.
        runtime mulai dihitung ketika kode yang berguna untuk mencari candidate
        words dieksekusi, dan runtime berhenti ketika baris code tersebut
        telah selesai dieksekusi
        """

        start_time = time.time()  # Waktu mulai

        # Best match accuracy
        best_match_accuracy = self.best_match_accuracy()

        # Candidate accuracy
        candidate_accuracy = self.candidate_accuracy()

        end_time = time.time()  # Waktu berakhir

        duration = end_time - start_time  # Selisih waktu mulai dan berkahirnya

        return {
            'cand_acc': candidate_accuracy,
            'best_acc': best_match_accuracy,
            'time': duration
        }

    def candidate_accuracy(self) -> float:
        """
        Menghitung candidate_accuracy dengan rumus m/n
        m = Jumlah instance dari correct word yang ada pada list of candidates.
        n = Total non-word errors yang ada pada test dataset.
        """
        m = 0
        n = 0
        for key, typo_list in self.__typo_sample_dataset:
            for typo in typo_list:

                # kata non-word errors kita cari kemungkinan kandidat kata yang benarnya
                candidates = self.__model.get_candidates(typo, 2)
                n += 1

                for candidate in candidates:

                    # Jika kata yang benar ditemukan pada kemungkinan kandidatnya
                    if (key == candidate):
                        m += 1

        return m / n

    def best_match_accuracy(self) -> float:
        """
        Menghitung best_match_accuracy dengan rumus m/n
        m = Jumlah instance dari correct word yang ada pada list of candidates 
        dan berada di posisi pertama pada list.
        n = Total non-word errors yang ada pada test dataset.
        """
        m = 0
        n = 0
        for key, typo_list in self.__typo_sample_dataset:
            for typo in typo_list:
                
                # kata non-word errors kita cari kemungkinan kandidat kata yang benarnya
                candidates = self.__model.get_candidates(typo, 2)
                n += 1

                # Jika kata yang benar ditemukan pada pada index pertama candidatesnya
                if (key == candidates[0]):
                    m += 1

        return m / n
