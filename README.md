## A Systematic Comparison of English Noun Compound Representations

This repository contains the code used in the paper "A Systematic Comparison of English Noun Compound Representations" in the MWE workshop @ ACL 2019. 
It can be used to train various types of noun compound embeddings and evaluate them on several task. 

### Dependencies

- Python 3
- argparse
- [allennlp (0.8.2)](https://github.com/allenai/allennlp/)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [word_forms](https://github.com/gutfeeling/word_forms)

### Training Noun Compound Embeddings

#### Noun Compound Vocabulary

The vocabulary is based on the [Tratz (2011) dataset](http://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/176191). It is found under `data/nc_vocab.txt`. This repository **does not** handle recognizing noun compounds.  

#### Corpus 

I used the English Wikipedia dump from January 2018 as the corpus. To download:

```
wget https://dumps.wikimedia.org/enwiki/20180101/enwiki-20180101-pages-meta-current.xml.bz2;
```

Then, we extract and clean the text using [WikiExtractor](https://github.com/attardi/wikiextractor):

```
python WikiExtractor.py --processes 20 -o corpora/text/ ~/corpora/[lang]wiki-20180101-pages-meta-current.xml.bz2;
cat corpora/text/*/* > output/en_corpus;
```

Finally, tokenize the corpus using [spacy](https://spacy.io/):

```
python training/distributional/preprocessing/tokenize_corpus.py output/en_corpus;
```

#### Training Noun Compound Embeddings

You can train embeddings with several training objectives:

- **Distributional** - learning vectors by treating noun compounds as single tokens. See [here](source/training/distributional/README.md). 
- **Compositional** - learning vectors by composing the embeddings of the constituent words. See [here](source/training/compositional/README.md). 
- **Paraphrase Based** - learning vectors by minimizing the distance to the embeddings of paraphrases. See [here](source/training/paraphrase_based/README.md). 

### Evaluation

The noun compound embeddings were evaluated on the following tasks:

- **Top K** - a qualitative analysis of the neighbours of each noun compound. See [here](source/evaluation/top_k/README.md). 
- **Classification** - multiclass classification of the noun compounds according to the relationship between the head and the modifier. See [here](source/evaluation/classification/README.md).
- **Attributes** - binary classification to whether a noun compound holds a certain property. See [here](source/evaluation/attributes/README.md).

### Citation

If you use this code for a scientific research, please cite the following paper:

```
@inproceedings{shwartz-2019-nc_embeddings,
    title = "A Systematic Comparison of English Noun Compound Representations",
    author = "Shwartz, Vered",
    booktitle = "Proceedings of the Joint Workshop on Multiword Expressions and WordNet (MWE-WN 2019) @ ACL 2019",
    month = August,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
}
```
