import os
import argparse

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from source.evaluation.common import load_text_embeddings
from source.evaluation.classification.dataset_reader import DatasetReader
from source.evaluation.classification.evaluation import evaluate, output_predictions


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('embedding_path', help='word embeddings to be used for w1 and w2 embeddings')
    ap.add_argument('embedding_dim', help='The embedding dimension', type=int)
    ap.add_argument('dataset_prefix', help='path to the train/test/val/rel data')
    ap.add_argument('model_dir', help='where to store the result')
    ap.add_argument('--is_compositional',
                    help='Whether the embeddings are from a compositional model', action='store_true')
    args = ap.parse_args()

    # Logging
    logdir = os.path.abspath(args.model_dir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler('{}/log.txt'.format(args.model_dir)),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    logger.info(f'Loading the datasets from {args.dataset_prefix}')
    train_set = DatasetReader(args.dataset_prefix + '/train.tsv')
    val_set = DatasetReader(args.dataset_prefix + '/val.tsv', label2index=train_set.label2index)
    test_set = DatasetReader(args.dataset_prefix + '/test.tsv', label2index=train_set.label2index)

    logger.info(f'Loading the embeddings from {args.embedding_path}')
    wv, index2word = load_text_embeddings(args.embedding_path, args.embedding_dim, normalize=True)
    word2index = {w: i for i, w in enumerate(index2word)}

    logger.info('Generating feature vectors...')
    prefix = 'comp_' if args.is_compositional else ''
    train_keys, test_keys, val_keys = [[prefix + '_'.join((w1, w2)) for w1, w2 in s.noun_compounds]
                                       for s in [train_set, test_set, val_set]]
    vocab = set(list(word2index.keys()))
    empty = np.zeros(args.embedding_dim)
    train_features, test_features, val_features = [np.vstack(
        [wv[word2index[key], :] if key in vocab else empty for key in s])
        for s in [train_keys, test_keys, val_keys]]

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

                logger.info('Training with classifier: {}, penalty: {}, c: {:.2f}...'.format(cls, penalty, reg_c))
                classifier.fit(train_features, train_set.labels)
                val_pred = classifier.predict(val_features)
                p, r, f1, _ = evaluate(val_set.labels, val_pred, val_set.index2label, do_full_reoprt=False)
                logger.info('Classifier: {}, penalty: {}, c: {:.2f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.
                            format(cls, penalty, reg_c, p, r, f1))
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
    precision, recall, f1, support = evaluate(test_set.labels, test_pred, test_set.index2label, do_full_reoprt=True)
    logger.info('Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(precision, recall, f1))

    # Write the predictions to a file
    output_predictions(args.model_dir + '/predictions.tsv', test_set.index2label, test_pred,
                       test_set.noun_compounds, test_set.labels)


if __name__ == '__main__':
    main()
