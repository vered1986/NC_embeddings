import random

random.seed(a=133)

import tqdm
import codecs
import logging
import argparse

import numpy as np

np.random.seed(133)

from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

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
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('dataset_file', help='path to the csv file')
    ap.add_argument('--is_compositional',
                    help='Whether the embeddings are from a compositional model', action='store_true')
    ap.add_argument('--is_paraphrase_based',
                    help='Whether the embeddings are from a paraphrase based model', action='store_true')
    args = ap.parse_args()

    if args.is_compositional and args.is_paraphrase_based:
        raise ValueError('Please select only one of args.is_compositional or args.is_paraphrase_based'
                         ' or none for distributional.')

    # Log
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    # Read the dataset
    logger.info('Reading the dataset from {}'.format(args.dataset_file))

    with codecs.open(args.dataset_file, 'r', 'utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
        dataset = {(w1, w2): tuple(map(float, (w1_score, w2_score, nc_score)))
                   for (w1, w2, w1_score, w2_score, nc_score) in lines}

    if args.is_compositional or args.is_paraphrase_based:
        logger.info(f'Loading model from {args.composition_model_path}')
        reader = NCDatasetReader() if args.is_compositional else NCParaphraseDatasetReader()
        archive = load_archive(args.composition_model_path)
        model = archive.model
        predictor = Predictor(model, dataset_reader=reader)

        logger.info('Computing vectors for the noun compounds')
        nc_to_vec = {}

        for w1, w2 in tqdm.tqdm(dataset.keys()):
            nc = '_'.join((w1, w2))
            instance = (nc, w1, w2) if args.is_compositional else (' '.join((w1, w2)), None, None)
            instance = reader.text_to_instance(*instance)

            if instance is None:
                logger.warning(f'Instance is None for {nc}')
            else:
                nc_to_vec[nc] = predictor.predict_instance(instance)['vector']

        logger.info('Training linear regression')

    # 3-fold cross validation, as in Reddy et al. (2011)
    ncs = list(dataset.keys())
    random.shuffle(ncs)
    fold_ncs = [ncs[i:i + len(ncs) // 3] for i in range(0, len(ncs), len(ncs) // 3)]
    regressor = LinearRegression()

    # (train_instances, train_gold, test_instances, test_gold)
    folds = [([(w1, w2) for index, fold in enumerate(fold_ncs) if index != test_index for (w1, w2) in fold],
              [dataset[(w1, w2)][2] for index, fold in enumerate(fold_ncs) if index != test_index for (w1, w2) in fold],
              fold_ncs[test_index],
              [dataset[(w1, w2)][2] for w1, w2 in fold_ncs[test_index]])
             for test_index in range(3)]

    folds = [(np.vstack([nc_to_vec['_'.join((w1, w2))] for w1, w2 in train_instances]), train_gold,
              np.vstack([nc_to_vec['_'.join((w1, w2))] for w1, w2 in test_instances]), test_gold)
             for train_instances, train_gold, test_instances, test_gold in folds]

    curr_scores = []

    for train_features, train_gold, test_features, test_gold in folds:
        regressor.fit(train_features, train_gold)
        test_predictions = regressor.predict(test_features)
        curr_scores.append(evaluate(test_gold, test_predictions))

    rhos, r_squares = zip(*curr_scores)
    print('rho = {:.3f}, r_squared = {:.3f}'.format(np.mean(rhos), np.mean(r_squares)))


def evaluate(y_test, y_pred):
    """
    Evaluate performance of the model on the test set
    :param y_test: the test set values
    :param y_pred: the predicted values
    :return: Spearman Rho and Goodness of fit R^2
    """
    rho = stats.spearmanr(y_test, y_pred)[0]
    rsquared = r2_score(y_test, y_pred)
    return rho, rsquared


if __name__ == '__main__':
    main()
