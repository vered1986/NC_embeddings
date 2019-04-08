import tqdm
import codecs
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

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    """
    Get a validation/test set, computes the compositional vectors of
    the noun compounds in the set, and saves the embeddings file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('nc_vocab', help='The noun compound vocabulary file')
    ap.add_argument('out_vector_file', help='Where to save the npy file')
    args = ap.parse_args()

    with codecs.open(args.nc_vocab, 'r', 'utf-8') as f_in:
        nc_vocab = [line.lower().replace('\t', '_') for line in f_in]

    logger.info(f'Loading model from {args.composition_model_path}')
    reader = NCDatasetReader()
    archive = load_archive(args.composition_model_path)
    model = archive.model
    predictor = Predictor(model, dataset_reader=reader)

    logger.info(f'Computing vectors for the noun compounds in {args.nc_vocab}')
    with codecs.open(args.out_vector_file, 'a', 'utf-8') as f_out:
        for nc in tqdm.tqdm(nc_vocab):
            w1, w2 = nc.split('_')
            instance = reader.text_to_instance(nc, w1, w2)

            if instance is None:
                logger.warning(f'Instance is None for {nc}')
            else:
                curr_vector = predictor.predict_instance(instance)['vector'][0]
                vector_text = ' '.join(map(str, curr_vector)).strip()
                f_out.write(f'comp_{nc} {vector_text}\n')


if __name__ == '__main__':
    main()
