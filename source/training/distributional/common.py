import codecs
import tarfile


def save_gensim_vectors(vectors, out_file, gzipped=True):
    """
    Save the Gensim word embeddings in GloVe textual format
    :param out_file: the txt file (will be saved as a gzipped file).
    """
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for word in vectors.wv.index2word:
            try:
                curr_vec = ' '.join(map(str, list(vectors[word])))
                f_out.write(' '.join((word, curr_vec)) + '\n')
            except:
                pass

    if gzipped:
        archive_file = out_file + '.gz'
        with tarfile.open(archive_file, 'w:gz') as archive:
            archive.add(out_file)
