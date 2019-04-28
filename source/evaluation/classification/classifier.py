import os
import re
import logging
import argparse

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from source.evaluation.compute_any_vector import compute_vectors
from source.evaluation.classification.dataset_reader import DatasetReader
from source.evaluation.classification.evaluation import evaluate, output_predictions


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('in_model_path', help='word embeddings or composition model path')
    ap.add_argument('dataset_prefix', help='path to the train/test/val/rel data')
    ap.add_argument('out_model_dir', help='where to store the result')
    args = ap.parse_args()

    # Logging
    logdir = os.path.abspath(args.out_model_dir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{args.out_model_dir}/log.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    exclude_labels = {'LEXICALIZED', 'PERSONAL_NAME', 'PERSONAL_TITLE'}

    logger.info(f'Loading the datasets from {args.dataset_prefix}')
    train_set = DatasetReader(args.dataset_prefix + '/train.tsv', exclude_labels=exclude_labels)
    val_set = DatasetReader(args.dataset_prefix + '/val.tsv', label2index=train_set.label2index,
                            exclude_labels=exclude_labels)
    test_set = DatasetReader(args.dataset_prefix + '/test.tsv', label2index=train_set.label2index,
                             exclude_labels=exclude_labels)

    # Compute the vectors for all the terms
    logger.info('Computing representations...')
    terms = train_set.noun_compounds + val_set.noun_compounds + test_set.noun_compounds
    term_to_vec = compute_vectors(args.in_model_path, terms)

    logger.info('Generating feature vectors...')
    embedding_dim = int(re.match('^.*/([0-9]+)d/.*$', args.out_model_dir).group(1))
    empty = np.zeros(embedding_dim)
    train_features, test_features, val_features = [np.vstack(
        [term_to_vec.get('_'.join(nc), empty) for nc in s.noun_compounds])
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

                logger.info(f'Training with classifier: {cls}, penalty: {penalty}, ' + 'c: {:.2f}'.format(reg_c))

                classifier.fit(train_features, train_set.labels)
                val_pred = classifier.predict(val_features)
                p, r, f1, _, _ = evaluate(val_set, val_pred)
                logger.info(f'Classifier: {cls}, penalty: {penalty}, ' +
                            'c: {:.2f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.format(reg_c, p, r, f1))
                f1_results.append(f1)
                models.append(classifier)

    best_index = np.argmax(f1_results)
    description = descriptions[best_index]
    classifier = models[best_index]
    logger.info(f'Best hyper-parameters: {description}')

    # Save the best model to a file
    logger.info('Copying the best model...')
    joblib.dump(classifier, f'{args.out_model_dir}/best.pkl')

    # Evaluate on the test set
    logger.info('Evaluation:')

    test_pred = classifier.predict(test_features)
    precision, recall, f1, support, full_report = evaluate(test_set, test_pred)
    logger.info('Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(precision, recall, f1))
    logger.info(full_report)

    # Write the predictions to a file
    output_predictions(args.out_model_dir + '/predictions.tsv', test_set.index2label, test_pred,
                       test_set.noun_compounds, test_set.labels)


if __name__ == '__main__':
    main()
