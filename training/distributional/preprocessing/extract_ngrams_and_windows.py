# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('corpus', help='The corpus file')
ap.add_argument('out_file', help='The output file')
ap.add_argument('nc_vocab', help='The vocabulary file')
args = ap.parse_args()

import string
import codecs
import itertools

from word_forms.word_forms import get_word_forms

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    logger.info('Reading the vocabulary')
    with codecs.open(args.nc_vocab, 'r', 'utf-8') as f_in:
        nc_vocab = frozenset([line.strip() for line in f_in])

    variations = {}
    for nc in nc_vocab:
        w1, w2 = nc.split('\t')
        variations[nc] = set(['_'.join((w1_form, w2_form)).lower()
                              for w1_form, w2_form in itertools.product(
                get_word_forms(w1)['n'], get_word_forms(w2)['n'])])

    logger.info('Processing...')
    with codecs.open(args.corpus, 'r', 'utf-8') as f_in:
        with codecs.open(args.out_file, 'w', 'utf-8') as f_out:
            try:
                for line in f_in:
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

    bigrams = enumerate(['_'.join((w1, w2)) for w1, w2 in zip(words, words[1:])])

    # Targets are bigrams, contexts are always unigrams
    for i, bigram in bigrams:
        nc = variations.get(bigram, None)
        if nc is not None:
            yield ' '.join(words[:i] + [nc] + words[i+2:]).strip()


if __name__ == '__main__':
    main()
