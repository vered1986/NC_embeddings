import tqdm
import codecs
import tarfile
import logging
import argparse

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

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
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('dataset', help='The dataset file')
    ap.add_argument('out_vector_file', help='Where to save the npy file')
    ap.add_argument('embedding_dim', type=int, help='The embedding dimension')
    args = ap.parse_args()

    logger.info(f'Loading model from {args.composition_model_path}')
    reader = NCDatasetReader()
    archive = load_archive(args.composition_model_path)
    model = archive.model
    predictor = Predictor(model, dataset_reader=reader)

    logger.info(f'Computing vectors for the noun compounds in {args.dataset}')

    with codecs.open(args.out_vector_file, 'w', 'utf-8') as f_out:
        with codecs.open(args.dataset, 'r', 'utf-8') as f_in:
            for line in tqdm.tqdm(f_in):
                nc = line.lower().replace('\t', '_')
                w1, w2 = nc.split('_')
                instance = reader.text_to_instance(nc, w1, w2)

                if instance is None:
                    logger.warning(f'Instance is None for {nc}')
                else:
                    curr_vector = predictor.predict_instance(instance)['vector']
                    f_out.write(nc + ' ' + ' '.join(map(str, list(curr_vector))) + '\n')

    archive_file = args.out_vector_file + '.gz'
    logger.info(f'Gzipping to {archive_file}')
    with tarfile.open(args.out_vector_file, 'w:gz') as archive:
        archive.add(args.out_vector_file)


if __name__ == '__main__':
    main()
