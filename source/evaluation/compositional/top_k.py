mport codecs
import logging
import argparse

import numpy as np

from source.evaluation.common import load_binary_embeddings, most_similar_word_by_vector, load_text_embeddings


logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('orig_emb_file', help='The gzipped text embedding file used by the model')
    ap.add_argument('nc_emb_file', help='The embedding .npy file computed with `compute_vectors.py`')
    ap.add_argument('nc_file', help='The test/val set file')
    args = ap.parse_args()

    logger.info(f'Loading distributional vectors from {args.orig_emb_file}')
    dist_wv, dist_index2word = load_text_embeddings(args.orig_emb_file, normalize=True)

    logger.info(f'Loading compositional vectors from {args.nc_emb_file}')
    comp_wv = load_binary_embeddings(args.nc_emb_file, normalize=True)

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        comp_index2word = [line.lower().replace('\t', '_') for line in f_in]

    logger.info('Uniting vectors')
    wv = np.vstack([dist_wv, comp_wv])
    index2word = [f'dist_{w}' for w in dist_index2word] + [f'comp_{w}' for w in comp_index2word]
    word2index = {w: i for i, w in enumerate(index2word)}

    for nc in comp_index2word:
        print(nc)
        vec = wv[word2index[f'comp_{nc}'], :]
        for other, score in most_similar_word_by_vector(index2word, wv, vec, k=20):
            print('\t'.join((other, '{:.3f}'.format(score))))

        print('')


if __name__ == '__main__':
    main()
