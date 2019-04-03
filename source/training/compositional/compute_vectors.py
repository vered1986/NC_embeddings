import tqdm
import codecs
import logging
import argparse

import numpy as np

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from source.training.compositional.nc_dataset_reader import NCDatasetReader

# For registration purposes - don't delete
from source.training.compositional.add_similarity import *
from source.training.compositional.composition_model import *
from source.training.compositional.matrix_similarity import *
from source.training.compositional.full_add_similarity import *

logger = logging.getLogger(__name__)


def main():
    """
    Get a validation/test set, computes the compositional vectors of
    the noun compounds in the set, and saves the embeddings file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('dataset', help='The dataset file')
    ap.add_argument('out_vector_file', help='Where to save the npy file')
    ap.add_argument('embedding_dim', type=int, help='The embedding dimension')
    args = ap.parse_args()

    logger.info(f'Loading model from {args.composition_model_path}')
    reader = NCDatasetReader()
    archive = load_archive(args.composition_model_path)
    model = archive.model
    predictor = Predictor(model, dataset_reader=reader)

    logger.info(f'Computing vectors for the noun compounds in {args.dataset}')
    vectors = []

    with codecs.open(args.dataset, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            nc = line.lower().replace('\t', '_')
            w1, w2 = nc.split('_')
            instance = reader.text_to_instance(nc, w1, w2)

            if instance is None:
                logger.warn(f'Instance is None for {nc}')
                curr_vector = np.zeros(args.embedding_dim)
            else:
                curr_vector = predictor.predict_instance(instance)['vector']

            vectors.append(curr_vector)

    logger.info(f'Saving vectors to {args.out_vector_file}')
    vectors = np.vstack(vectors)
    np.save(args.out_vector_file, vectors)


if __name__ == '__main__':
    main()
