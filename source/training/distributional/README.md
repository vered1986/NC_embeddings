# Distributional Embeddings

Learning vectors for words and noun compounds based 
on their occurrences in the corpus.

The steps are as follows:

## 1. Preparing the Compound-Aware Corpus

The following scripts create a version of the corpus that treats 
noun compounds as single tokens, 
considering also inflectional variants such as plurality: 

```
python -m training/distributional/preprocessing/extract_ngrams_and_windows output/en_corpus data/nc_vocab.txt
```

## 2. Training Embeddings

* [word2vec](word2vec/train_all.sh): using the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) implementation
* [GloVe](glove/train_all.sh): using the [original](https://github.com/stanfordnlp/GloVe) implementation 
* [FastText](fasttext/train_all.sh): using the [gensim](https://radimrehurek.com/gensim/models/fasttext.html) implementation

