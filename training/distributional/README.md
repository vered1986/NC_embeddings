# Distributional Embeddings

Learning vectors for words and noun compounds based 
on their occurrences in the corpus.

The steps are as follows:

## 1. Getting the Wikipedia Corpus

We use the English Wikipedia dump from January 2018. To download:

```
wget https://dumps.wikimedia.org/enwiki/20180101/enwiki-20180101-pages-meta-current.xml.bz2;
```

Then, we extract and clean the text using [WikiExtractor](https://github.com/attardi/wikiextractor):

```
python WikiExtractor.py --processes 20 -o corpora/text/ ~/corpora/[lang]wiki-20180101-pages-meta-current.xml.bz2;
cat corpora/text/*/* > output/en_corpus;
```

Finally, we tokenize the corpus using [spacy](https://spacy.io/):

```
python training/distributional/preprocessing/tokenize_corpus.py output/en_corpus;
```

## 2. Preparing the Compound-Aware Corpus

We will base our NC vocabulary on the [Tratz (2011) dataset](http://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/176191). 

The following scripts create a version of the corpus that treats 
noun compounds as single tokens, 
considering also inflectional variants such as plurality: 

```
python training/distributional/preprocessing/extract_ngrams_and_windows.py \
        output/en_corpus data/nc_vocab.txt; 
```

## 3. Training Embeddings

* [word2vec](word2vec/train_all.sh): using the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) implementation
* [GloVe](glove/train_all.sh): using the [original](https://github.com/stanfordnlp/GloVe) implementation 
* [FastText](fasttext/train_all.sh): using the [gensim](https://radimrehurek.com/gensim/models/fasttext.html) implementation

