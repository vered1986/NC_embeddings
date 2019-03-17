# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--iter', type=int, help='Number of epochs', default=5)
ap.add_argument('--window_size', help='Context window size', type=int, default=2)
ap.add_argument('--embedding_dim', help='Embeddings dimension', type=int, default=100)
ap.add_argument('--workers', help='Number of parallel workers', type=int, default=16)
ap.add_argument('--algorithm', help='Skip-gram (sg) or CBOW (cbow)', type=str, default='sg')
ap.add_argument('--min_count', help='Minimum target word occurrences in the corpus', type=int, default=1)
ap.add_argument('corpus', help='The corpus file')
ap.add_argument('output_dir', help='Where to store the word embeddings files (.npy and .vocab)')
args = ap.parse_args()

# Log
import os
logdir = os.path.abspath(args.output_dir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

import logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('{}/log.txt'.format(logdir)),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

import gensim

from source.training.distributional.corpus_reader import CorpusReader
from source.training.distributional.common import save_gensim_vectors


def main():
    sentence_iter = CorpusReader(args.corpus)

    logger.info('Training...')
    try:
        model = gensim.models.FastText(sentence_iter,
                                       size=args.embedding_dim,
                                       window=args.window_size,
                                       min_count=args.min_count,
                                       sg=1 if args.algorithm.lower() == 'sg' else 0,
                                       workers=args.workers,
                                       iter=args.iter)
    except Exception as err:
        logger.error(f'Failed to train model. Exiting now.\n{err}')
        return 0

    # Save
    try:
        logger.info(f'Saving the embeddings to {args.output_dir}')
        model.save(args.output_dir + '/wv.bin')
        save_gensim_vectors(model, os.path.join(args.output_dir, '/embeddings'))
    except Exception as err:
        logger.error(f'Failed to save model.\n{err}')


if __name__ == '__main__':
    main()
