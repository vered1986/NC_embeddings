import re
import tqdm
import codecs
import logging
import argparse

import numpy as np

np.random.seed(133)

from scipy import stats
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

# For registration purposes - don't delete
from source.training.compositional.add_similarity import *
from source.training.compositional.composition_model import *
from source.training.compositional.matrix_similarity import *
from source.training.compositional.nc_dataset_reader import *
from source.training.compositional.full_add_similarity import *
from source.training.paraphrase_based.paraphrase_composition_model import *
from source.training.paraphrase_based.nc_paraphrases_dataset_reader import *


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('model_paths', help='## separated list of model.tar.gz paths')
    ap.add_argument('dataset_file', help='path to the csv file')
    args = ap.parse_args()

    # Log
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    # Read the dataset
    logger.info('Reading the dataset from {}'.format(args.dataset_file))

    with codecs.open(args.dataset_file, 'r', 'utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
        dataset = {'_'.join((w1, w2)): float(nc_score) for (w1, w2, w1_score, w2_score, nc_score) in lines}

    nc_to_vec = defaultdict(list)

    model_paths = args.model_paths.split('##')

    for model_path in model_paths:
        is_compositional = 'compositional' in model_path
        model_type = 'compositional' if is_compositional else 'paraphrase based'
        emb_dim = int(re.match('^.*/([0-9]+)d/.*$', model_path).group(1))
        logger.info(f'Loading model from {model_path}, model type: {model_type}, dimension: {emb_dim}')

        reader = NCDatasetReader() if is_compositional else NCParaphraseDatasetReader()
        archive = load_archive(model_path)
        model = archive.model
        predictor = Predictor(model, dataset_reader=reader)

        logger.info('Computing vectors for the noun compounds')

        for nc in tqdm.tqdm(dataset.keys()):
            w1, w2 = nc.split('_')
            instance = (nc, w1, w2) if 'compositional' in model_path else (' '.join((w1, w2)), None, None)
            instance = reader.text_to_instance(*instance)

            if instance is None:
                logger.warning(f'Instance is None for {nc}, adding zero vector instead')
                nc_to_vec[nc].append(np.zeros(emb_dim))
            else:
                nc_to_vec[nc].append(np.array(predictor.predict_instance(instance)['vector']).squeeze())

    assert all([len(vectors) == len(model_paths) for vectors in nc_to_vec.values()]), 'not all NCs have all vectors'
    assert all([len({v.shape for v in vectors}) == 1 for vectors in nc_to_vec.values()]), 'vectors with different shapes'

    # Define the dataset
    X, y = zip(*[(np.concatenate(vectors, axis=-1), dataset[nc]) for nc, vectors in nc_to_vec.items()])
    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)

    # Train the regression and evaluate using 3-fold cross validation, as in Reddy et al. (2011)
    logger.info('Training linear regression')

    def rho_score(y, y_pred, **kwargs):
        return stats.spearmanr(y, y_pred)[0]

    scoring = {'r_squared': 'r2', 'rho': make_scorer(rho_score)}
    scores = cross_validate(Ridge(alpha=5), X, y, cv=3, scoring=scoring)
    print('rho = {:.3f}, r_squared = {:.3f}'.format(np.mean(scores['test_rho']), np.mean(scores['test_r_squared'])))


if __name__ == '__main__':
    main()
