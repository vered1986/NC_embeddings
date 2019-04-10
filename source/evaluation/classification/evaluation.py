import codecs

from sklearn import metrics


def output_predictions(predictions_file, relations, predictions, test_set_keys, test_labels):
    """
    Output the model predictions for the test set
    :param predictions_file: the output file path
    :param relations: the ordered list of relations
    :param predictions: the predicted labels for the test set
    :param test_set: the test set - a list of (w1, w2, relation) instances
    :return:
    """
    with codecs.open(predictions_file, 'w', 'utf-8') as f_out:
        for i, (w1, w2) in enumerate(test_set_keys):
            f_out.write('\t'.join([w1, w2, relations[test_labels[i]], relations[predictions[i]]]) + '\n')


def evaluate(y_test, y_pred, relations):
    """
    Evaluate performance of the model on the test set
    :param y_test: the test set labels.
    :param y_pred: the predicted values
    :return: mean F1 over all classes
    """
    full_report = metrics.classification_report(y_test, y_pred, target_names=relations, digits=3)
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return pre, rec, f1, support, full_report

