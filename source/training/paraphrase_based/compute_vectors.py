import tqdm
import codecs
import logging
import argparse

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

# For registration purposes - don't delete
from source.training.paraphrase_based.paraphrase_composition_model import *
from source.training.paraphrase_based.nc_paraphrases_dataset_reader import NCParaphraseDatasetReader

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


from typing import Dict
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import LabelField, TextField, IndexField, SpanField


@DatasetReader.register("nc_paraphrases_data_reader_for_words")
class NCParaphraseDatasetReaderForWords(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def _read(self, file_path):
        pass

    @overrides
    def text_to_instance(self, nc: str) -> Instance:
        tokenized_nc = self._tokenizer.tokenize(nc)
        nc_field = TextField(tokenized_nc, self._token_indexers)
        fields = {'nc': nc_field}
        return Instance(fields)


def main():
    """
    Get a validation/test set, computes the compositional vectors of
    the noun compounds in the set, and saves the embeddings file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('composition_model_path', help='The composition model file (model.tar.gz)')
    ap.add_argument('nc_vocab', help='The noun compound vocabulary file')
    ap.add_argument('vocab', help='The word vocabulary file')
    ap.add_argument('out_vector_file', help='Where to save the gzipped file')
    args = ap.parse_args()

    with codecs.open(args.nc_vocab, 'r', 'utf-8') as f_in:
        nc_vocab = [line.strip().lower().replace('\t', ' ') for line in f_in]

    with codecs.open(args.vocab, 'r', 'utf-8') as f_in:
        vocab = [line.strip().lower().replace('\t', ' ') for line in f_in]

    vocab += ['_'.join(nc.split()) for nc in nc_vocab if len(nc.split()) == 2]

    logger.info(f'Loading model from {args.composition_model_path}')
    archive = load_archive(args.composition_model_path)
    model = archive.model

    with codecs.open(args.out_vector_file, 'a', 'utf-8') as f_out:
        logger.info(f'Computing vectors for the single words in {args.vocab}')
        reader = NCParaphraseDatasetReaderForWords()
        predictor = Predictor(model, dataset_reader=reader)

        for word in tqdm.tqdm(vocab):
            instance = reader.text_to_instance(word)

            if instance is None:
                logger.warning(f'Instance is None for {word}')
            else:
                curr_vector = predictor.predict_instance(instance)['vector']

                if len(curr_vector) == 1:
                    curr_vector = curr_vector[0]

                vector_text = ' '.join(map(str, curr_vector)).strip()
                f_out.write(f'comp_{word} {vector_text}\n')

        logger.info(f'Computing vectors for the noun compounds in {args.nc_vocab}')
        reader = NCParaphraseDatasetReader()
        for nc in tqdm.tqdm(nc_vocab):
            instance = reader.text_to_instance(nc)

            if instance is None:
                logger.warning(f'Instance is None for {nc}')
            else:
                curr_vector = predictor.predict_instance(instance)['vector']

                if len(curr_vector) == 1:
                    curr_vector = curr_vector[0]

                vector_text = ' '.join(map(str, curr_vector)).strip()
                nc = nc.replace(' ', '_')
                f_out.write(f'dist_{nc} {vector_text}\n')


if __name__ == '__main__':
    main()

