# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('corpus', help='The corpus file')
ap.add_argument('out_file', help='The output file')
ap.add_argument('nc_vocab', help='The vocabulary file')
args = ap.parse_args()

import os
import tqdm
import string
import codecs
import itertools
import subprocess

from word_forms.word_forms import get_word_forms

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    logger.info('Reading the vocabulary')
    with codecs.open(args.nc_vocab, 'r', 'utf-8') as f_in:
        nc_vocab = frozenset([line.strip() for line in f_in])

    logger.info('Computing variations...')
    variations = {}
    for nc in nc_vocab:
        w1, w2 = nc.split('\t')
        variations[nc.replace('\t', '_')] = \
            set(['_'.join((w1_form, w2_form)).lower()
                 for w1_form, w2_form in itertools.product(
                    get_word_forms(w1)['n'], get_word_forms(w2)['n'])])

    logger.info('Counting the number of sentences in the corpus')
    num_instances = corpus_size(args.corpus)

    logger.info('Processing...')
    with codecs.open(args.corpus, 'r', 'utf-8') as f_in:
        with codecs.open(args.out_file, 'w', 'utf-8') as f_out:
            try:
                for line in tqdm.tqdm(f_in, total=num_instances):
                    for sentence in get_sentences_with_bigrams(line.strip().lower(), variations):
                        f_out.write(sentence + '\n')

            except Exception as err:
                logger.error(err)


def get_sentences_with_bigrams(sentence, variations):
    """
    Returns all the possible splits to bigrams
    :return: a list of sentences where bigrams appear as a single token with underscores
    """
    words = [w for w in sentence.split() if w not in string.punctuation]

    # Original sentence
    yield ' '.join(words)

    bigrams = ['_'.join((w1, w2)) for w1, w2 in zip(words, words[1:])]

    # Targets are bigrams, contexts are always unigrams
    for i, bigram in enumerate(bigrams):
        nc = variations.get(bigram, None)
        if nc is not None:
            yield ' '.join(words[:i] + [nc] + words[i+2:]).strip()


def corpus_size(corpus_file):
    """
    Count the number of distinct word pairs in the corpus, or read it from the cached file
    """
    if os.path.exists(corpus_file + '_num_instances'):
        with open(corpus_file + '_num_instances') as f_in:
            num_instances = int(f_in.read().strip())

    else:
        num_instances = None
        p = subprocess.Popen(['wc', '-l', corpus_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode == 0:
            num_instances = int(result.strip().split()[0])

        with open(corpus_file + '_num_instances', 'w') as f_out:
            f_out.write(str(num_instances) + '\n')

    return num_instances


if __name__ == '__main__':
    main()
