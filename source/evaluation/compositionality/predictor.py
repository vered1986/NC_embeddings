import random
random.seed(a=133)

import codecs
import logging
import argparse

import numpy as np
np.random.seed(133)

from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from source.evaluation.common import load_text_embeddings


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('embedding_path', help='word embeddings to be used for w1 and w2 embeddings')
    ap.add_argument('dataset_file', help='path to the csv file')
    ap.add_argument('--is_compositional',
                    help='Whether the embeddings are from a compositional model', action='store_true')
    args = ap.parse_args()

    # Log
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    # Read the dataset
    logger.info('Reading the dataset from {}'.format(args.dataset_file))

    with codecs.open(args.dataset_file, 'r', 'utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
        dataset = { (w1, w2) : tuple(map(float, (w1_score, w2_score, nc_score)))
                    for (w1, w2, w1_score, w2_score, nc_score) in lines }

    logger.info(f'Loading the embeddings from {args.embedding_path}')
    wv, index2word = load_text_embeddings(args.emb_file, args.embedding_dim, normalize=True)
    word2index = {w: i for i, w in enumerate(index2word)}

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

    prefix = 'comp_' if args.is_compositional else ''
    folds = [(np.vstack([wv[word2index[prefix + '_'.join((w1, w2))], :] for w1, w2 in train_instances]),
              train_gold,
              np.vstack([wv[word2index[prefix + '_'.join((w1, w2))], :] for w1, w2 in test_instances]),
              test_gold)
             for train_instances, train_gold, test_instances, test_gold in folds]

    curr_scores = []

    for train_features, train_gold, test_features, test_gold in folds:
        regressor.fit(train_features, train_gold)
        test_predictions = regressor.predict(test_features)
        curr_scores.append(evaluate(test_gold, test_predictions))

    rhos, r_squares = zip(*curr_scores)
    logger.info('rho = {:.3f}, r_squared = {:.3f}'.format(np.mean(rhos), np.mean(r_squares)))


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
