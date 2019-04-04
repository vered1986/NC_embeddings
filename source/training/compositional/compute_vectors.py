import tqdm
import codecs
import tarfile
import logging
import argparse

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from source.evaluation.common import load_text_embeddings
from source.training.compositional.nc_dataset_reader import NCDatasetReader

# For registration purposes - don't delete
from source.training.compositional.add_similarity import *
from source.training.compositional.composition_model import *
from source.training.compositional.matrix_similarity import *
from source.training.compositional.full_add_similarity import *

logger = logging.getLogger(__name__)


def main():
    """
    Get a validation/test set, computes the compositional vectors of
    the noun compounds in the set, and saves the embeddings file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('orig_emb_file', help='The gzipped text embedding file used by the model')
    ap.add_argument('embedding_dim', help='The embedding dimension', type=int)
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('nc_vocab', help='The noun compound vocabulary file')
    ap.add_argument('out_vector_file', help='Where to save the npy file')
    args = ap.parse_args()

    logger.info(f'Loading distributional vectors from {args.orig_emb_file}')
    dist_wv, dist_index2word = load_text_embeddings(args.orig_emb_file, args.embedding_dim)

    logger.info(f'Loading model from {args.composition_model_path}')
    reader = NCDatasetReader()
    archive = load_archive(args.composition_model_path)
    model = archive.model
    predictor = Predictor(model, dataset_reader=reader)

    logger.info(f'Computing vectors for the noun compounds in {args.dataset}')
    comp_wv, comp_index2word = [], []

    with codecs.open(args.nc_vocab, 'r', 'utf-8') as f_in:
        nc_vocab = [line.lower().replace('\t', '_') for line in f_in]

    for nc in tqdm.tqdm(nc_vocab):
        w1, w2 = nc.split('_')
        instance = reader.text_to_instance(nc, w1, w2)

        if instance is None:
            logger.warning(f'Instance is None for {nc}')
        else:
            curr_vector = predictor.predict_instance(instance)['vector']
            comp_index2word.append(nc)
            comp_wv.append(curr_vector)

    logger.info('Uniting vectors')
    wv = dist_wv + comp_wv
    index2word = [f'dist_{w}' for w in dist_index2word] + [f'comp_{w}' for w in comp_index2word]
    word2index = {w: i for i, w in enumerate(index2word)}

    logger.info(f'Writing to {args.out_vector_file}')
    with codecs.open(args.out_vector_file, 'w', 'utf-8') as f_out:
        for word, curr_vector in zip(word2index, wv):
            f_out.write(word + ' ' + ' '.join(map(str, list(curr_vector))) + '\n')

    archive_file = args.out_vector_file + '.gz'
    logger.info(f'Gzipping to {archive_file}')
    with tarfile.open(args.out_vector_file, 'w:gz') as archive:
        archive.add(args.out_vector_file)


if __name__ == '__main__':
    main()
