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



def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
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

    logger.info(f'Loading model from {args.composition_model_path}')
    archive = load_archive(args.composition_model_path)
    model = archive.model
    predictor = Predictor(model)

    logger.info('Computing vectors for the noun compounds')
    nc_to_vec = {}

    for nc in tqdm.tqdm(dataset.keys()):
        w1, w2 = nc.split('_')
        instance = text_to_instance(nc, w1, w2)

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


def text_to_instance(nc: str, w1: str, w2: str):
    tokenizer = WordTokenizer()
    token_indexers = {"tokens": SingleIdTokenIndexer()}
    tokenized_nc = tokenizer.tokenize(nc)
    nc_field = TextField(tokenized_nc, token_indexers)

    # Remove non-binary NCs
    if nc_field.sequence_length() != 1:
        return None

    tokenized_w1 = tokenizer.tokenize(w1)
    w1_field = TextField(tokenized_w1, token_indexers)
    tokenized_w2 = tokenizer.tokenize(w2)
    w2_field = TextField(tokenized_w2, token_indexers)
    tokenized_nc_seq = tokenizer.tokenize(' '.join((w1, w2)))
    nc_seq_field = TextField(tokenized_nc_seq, token_indexers)

    fields = {'nc': nc_field, 'w1': w1_field, 'w2': w2_field, 'nc_seq': nc_seq_field}
    return Instance(fields)


if __name__ == '__main__':
    main()