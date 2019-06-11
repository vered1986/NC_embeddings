import codecs
import logging
import argparse

from source.evaluation.common import load_text_embeddings, most_similar_word


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('emb_file', help='The gzipped text embedding file')
    ap.add_argument('embedding_dim', help='The embedding dimension', type=int)
    ap.add_argument('nc_file', help='The test/val set file')
    ap.add_argument('--is_compositional',
                    help='Whether the embeddings are from a compositional/paraphrase-based model', action='store_true')
    args = ap.parse_args()

    logger.info(f'Loading vectors from {args.emb_file}')
    wv, index2word = load_text_embeddings(args.emb_file, args.embedding_dim, normalize=True)
    word2index = {w: i for i, w in enumerate(index2word)}

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        ncs = [line.strip().lower().replace('\t', '_') for line in f_in]

    # Add "comp_" before each NC
    if args.is_compositional:
        ncs = [f'comp_{nc}' for nc in ncs]

    for nc in ncs:
        print(nc)
        for other, score in most_similar_word(index2word, word2index, wv, nc, k=20):
            print('\t'.join((other, '{:.3f}'.format(score))))

        print('')


if __name__ == '__main__':
    main()
