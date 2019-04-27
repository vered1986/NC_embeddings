import codecs


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
        for i, term in enumerate(test_set_keys):
            f_out.write('\t'.join([term, relations[test_labels[i]], relations[predictions[i]]]) + '\n')
