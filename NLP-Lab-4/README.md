# NLP - Lab Task 4
Develop N-gram Language Model for NLP - Lab Task 4

## Disclaimer
This README.md file as supplementary material to the problem document. You can refer directly to the problem document for a comprehensive and detailed explanation.

Each TODO program has a comment above it so that you can understand easily what to do and how to solve it.

### How to get the dataset
You can access it straight to the folder data. Three different text files exist, i.e., idwiki-corpus, idwiki-train, and idwiki-test. The idwiki-corpus.txt consists of 100 sentences already split with an 80:20 ratio to the idwiki-train and idwiki-test.

### How to tokenize the sentence using Aksara
You can refer to the [repository - ir-nlp-csui/aksara](https://github.com/ir-nlp-csui/aksara/blob/main/docs/source/user_guide/tokenize_text.rst)

### Prerequisite
- Python version **must be** < 3.11 due to aksara package does not support Python version $\geq$ 3.11.

### How to start
```
python -m venv env
source env/Scripts/activate

pip install -r requirements.txt
python main.py
```

### How to debug the python file
Uncomment the conditional statement
```
if __name__ == "__main__":
  main()
```
then run the python file
```
python [filename].py
```

### References and Supplements
1. [Scele NLP - Semester Gasal 2024/2025](https://scele.cs.ui.ac.id/pluginfile.php/238150/mod_resource/content/1/04%20Language%20Modeling.pdf)
2. [Aksara](https://github.com/ir-nlp-csui/aksara)
3. [Speech and Language Processing (3rd ed. draft) - N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
