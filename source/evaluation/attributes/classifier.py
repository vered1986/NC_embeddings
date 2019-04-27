import re
import os
import logging
import argparse

import numpy as np

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from source.evaluation.attributes.dataset_reader import DatasetReader
from source.evaluation.attributes.evaluation import output_predictions
from source.evaluation.attributes.compute_any_vector import compute_vectors


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path', help='word embeddings or composition model path')
    ap.add_argument('dataset_prefix', help='path to the train/test/val/rel data')
    ap.add_argument('model_dir', help='where to store the result')
    args = ap.parse_args()

    # Logging
    logdir = os.path.abspath(args.model_dir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{args.model_dir}/log.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info(f'Loading the datasets from {args.dataset_prefix}')
    train_set = DatasetReader(args.dataset_prefix + '_train.tsv', label2index={'False': 0, 'True': 1})
    val_set = DatasetReader(args.dataset_prefix + '_val.tsv', label2index=train_set.label2index)
    test_set = DatasetReader(args.dataset_prefix + '_test.tsv', label2index=train_set.label2index)

    # Compute the vectors for all the terms
    logger.info(f'Computing representations...')
    terms = train_set.noun_compounds + val_set.noun_compounds + test_set.noun_compounds
    term_to_vec = compute_vectors(args.model_path, terms)

    logger.info('Generating feature vectors...')
    embedding_dim = int(re.match('^.*/([0-9]+)d/.*$', args.model_path).group(1))
    empty = np.zeros(embedding_dim)
    train_features, test_features, val_features = [np.vstack(
        [term_to_vec.get(term, empty) for term in s.noun_compounds])
        for s in [train_set, test_set, val_set]]

    # Tune the hyper-parameters using the validation set
    logger.info('Classifying...')
    reg_values = [0.5, 1, 2, 5, 10]
    penalties = ['l2']
    classifiers = ['logistic', 'svm']
    f1_results = []
    descriptions = []
    models = []

    for cls in classifiers:
        for reg_c in reg_values:
            for penalty in penalties:
                descriptions.append(f'Classifier: {cls}, Penalty: {penalty}, C: {reg_c}')

                # Create the classifier
                if cls == 'logistic':
                    classifier = LogisticRegression(penalty=penalty, C=reg_c,
                                                    multi_class='multinomial', n_jobs=20, solver='sag')
                else:
                    classifier = LinearSVC(penalty=penalty, dual=False, C=reg_c)

                logger.info('Training with classifier: {}, penalty: {}, c: {:.2f}, ...'.format(cls, penalty, reg_c))

                classifier.fit(train_features, train_set.labels)
                val_pred = classifier.predict(val_features)
                p, r, f1, _ = metrics.precision_recall_fscore_support(val_set.labels, val_pred,
                                                                      pos_label=1, average='binary')
                logger.info(f'Classifier: {cls}, penalty: {penalty}, ' +
                            'c: {:.2f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.
                            format(reg_c, p, r, f1))
                f1_results.append(f1)
                models.append(classifier)

    best_index = np.argmax(f1_results)
    description = descriptions[best_index]
    classifier = models[best_index]
    logger.info(f'Best hyper-parameters: {description}')

    # Save the best model to a file
    logger.info('Copying the best model...')
    joblib.dump(classifier, f'{args.model_dir}/best.pkl')

    # Evaluate on the test set
    logger.info('Evaluation:')

    test_pred = classifier.predict(test_features)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(test_set.labels, test_pred,
                                                                             pos_label=1, average='binary')
    logger.info('Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(precision, recall, f1))

    # Write the predictions to a file
    output_predictions(args.model_dir + '/predictions.tsv', test_set.index2label, test_pred,
                       test_set.noun_compounds, test_set.labels)


if __name__ == '__main__':
    main()
