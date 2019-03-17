import gzip


def save_gensim_vectors(vectors, out_file):
    """
    Save the Gensim word embeddings in GloVe textual format
    :param out_file: the txt file (will be saved as a gzipped file).
    """
    with gzip.open(out_file, 'wb', 'utf-8') as f_out:
        for word in vectors.wv.index2word:
            try:
                curr_vec = ' '.join(map(str, list(vectors[word])))
                f_out.write(' '.join((word, curr_vec)) + '\n')
            except:
                pass
