import gzip
import codecs
import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_binary_embeddings(embeddings_file, normalize=False):
    """
    Load binary word embeddings
    :param embeddings_file: the npy file with the matrix
    :return: the matrix
    """
    wv = np.load(embeddings_file)

    # Normalize each row (word vector) in the matrix to sum-up to 1
    if normalize:
        row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
        wv /= row_norm[:, np.newaxis]

    return wv


def load_text_embeddings(file_name, normalize=False):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :return: the vocabulary and the word vectors
    """
    with codecs.getreader('utf-8')(gzip.open(file_name), 'rb', errors='replace') as f_in:
        lines = [line.strip() for line in f_in]

    embedding_dim = len(lines[0].split()) - 1
    words, vectors = zip(*[line.strip().split(' ', 1)
                           for line in lines
                           if len(line.split()) == embedding_dim + 1])
    wv = np.loadtxt(vectors)

    # Normalize each row (word vector) in the matrix to sum-up to 1
    if normalize:
        row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
        wv /= row_norm[:, np.newaxis]

    return wv, words


def most_similar_word(index2word, word2index, wv, w, k=10):
    """
    Returns the k most similar words to w
    :param index2word: the list of vocabulary words
    :param word2index: a word to index dictionary
    :param wv: the word embeddings
    :param w: the target word
    :param k: the number of words to return
    :return: the k most similar words to w
    """
    index = word2index.get(w, -1)

    if index < 0:
        return []

    # Apply matrix-vector dot product to get the distance of w from all the other vectors
    distance = np.dot(wv, wv[index, :])

    max_indices = (-distance).argsort()[:k + 1]
    words = [(index2word[i], distance[i]) for i in max_indices if i != index]

    return words
