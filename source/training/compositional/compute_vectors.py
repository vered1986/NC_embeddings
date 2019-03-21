import json
import tqdm
import codecs
import logging
import argparse

import numpy as np

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from source.training.compositional.nc_dataset_reader import NCDatasetReader

logger = logging.getLogger(__name__)


def main():
    """
    Get a validation/test set, computes the compositional vectors of
    the noun compounds in the set, and saves the embeddings file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('dataset', help='The dataset jsonl file')
    ap.add_argument('out_vector_file', help='Where to save the npy file')
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
            example = json.loads(line.strip())
            vectors.append(predictor.predict(example['nc'])['vector'])

    logger.info(f'Saving vectors to {args.out_vector_file}')
    vectors = np.vstack(vectors)
    np.save(args.out_vector_file, vectors)


if __name__ == '__main__':
    main()
