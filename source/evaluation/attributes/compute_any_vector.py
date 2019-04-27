import re
import tqdm
import logging

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

# For registration purposes - don't delete
from allennlp.models.model import Model
from source.evaluation.common import load_text_embeddings
from source.training.compositional.add_similarity import *
from source.training.compositional.composition_model import *
from source.training.compositional.matrix_similarity import *
from source.training.compositional.full_add_similarity import *
from source.training.paraphrase_based.paraphrase_composition_model import *
from source.training.compositional.composition_model import CompositionModel

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

from typing import Dict
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import LabelField, TextField, IndexField, SpanField


@DatasetReader.register("nc_paraphrases_data_reader_single_words")
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
        nc = nc.replace('_', ' ')
        tokenized_nc = self._tokenizer.tokenize(nc)
        nc_field = TextField(tokenized_nc, self._token_indexers)
        fields = {'nc': nc_field}
        return Instance(fields)


@DatasetReader.register("nc_data_reader_single_words")
class NCDatasetReaderForWords(DatasetReader):
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
        w1_field, w2_field, nc_seq_field = nc_field, nc_field, nc_field

        constituents = nc.split('_')
        if len(constituents) == 2:
            w1, w2 = constituents
            tokenized_w1 = self._tokenizer.tokenize(w1)
            w1_field = TextField(tokenized_w1, self._token_indexers)
            tokenized_w2 = self._tokenizer.tokenize(w2)
            w2_field = TextField(tokenized_w2, self._token_indexers)
            tokenized_nc_seq = self._tokenizer.tokenize(' '.join((w1, w2)))
            nc_seq_field = TextField(tokenized_nc_seq, self._token_indexers)

        fields = {'nc': nc_field, 'w1': w1_field, 'w2': w2_field, 'nc_seq': nc_seq_field}
        return Instance(fields)


@Model.register("single_word_composition_model")
class SingleWordCompositionModel(CompositionModel):
    def __init__(self, model):
        self.__dict__ = model.__dict__

    @overrides
    def forward(self,  # type: ignore
                nc: Dict[str, torch.LongTensor],
                w1: Dict[str, torch.LongTensor],
                w2: Dict[str, torch.LongTensor],
                nc_seq: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        term_emb = self.text_field_embedder(nc)
        output_dict = {'vector': term_emb}
        return output_dict


def compute_vectors(model_path, terms):
    """
    Compute the vectors of the terms using the model / pre-trained distributional embeddings
    :param model_path: the path to the pre-trained composition model / pre-trained distributional embeddings
    :param terms: a list of terms (noun compounds and single words) to compute
    :return: a dictionary of term to vector
    """
    # Embeddings
    if model_path.endswith('embeddings.txt.gz'):
        logger.info(f'Loading the embeddings from {model_path}')
        embedding_dim = int(re.match('^.*/([0-9]+)d/.*$', model_path).group(1))
        wv, index2word = load_text_embeddings(model_path, embedding_dim)
        word2index = {w: i for i, w in enumerate(index2word)}
        vocab = set(list(word2index.keys()))
        term_to_vec = {term: wv[word2index[term], :] for term in tqdm.tqdm(terms) if term in vocab}
        return term_to_vec

    # Paraphrase based
    elif 'paraphrase_based' in model_path:
        return compute_compositional_vectors(model_path, terms, nc_reader=NCParaphraseDatasetReaderForWords(),
                                             single_word_reader=NCParaphraseDatasetReaderForWords())
    # Compositional
    else:
        return compute_compositional_vectors(model_path, terms,
                                             single_word_composition_model=SingleWordCompositionModel)


def compute_compositional_vectors(model_path, terms, nc_reader=NCDatasetReaderForWords(),
                                  single_word_reader=NCDatasetReaderForWords(),
                                  single_word_composition_model=None):
    """
    Compute the vectors of the terms using the model
    :param model_path: the path to the pre-trained composition model
    :param terms: a list of terms (noun compounds and single words) to compute
    :return: a dictionary of term to vector
    """
    logger.info(f'Loading model from {model_path}')
    archive = load_archive(model_path)
    model = archive.model
    predictor = Predictor(model, dataset_reader=single_word_reader)

    single_word_predictor = predictor
    if single_word_composition_model:
        single_word_model = single_word_composition_model(model)
        single_word_predictor = Predictor(single_word_model, dataset_reader=single_word_reader)

    term_to_vec = {}

    for term in tqdm.tqdm(terms):
        if '_' in term:
            curr_predictor = predictor
            curr_reader = nc_reader
        else:
            curr_predictor = single_word_predictor
            curr_reader = single_word_reader

        instance = curr_reader.text_to_instance(term)

        if instance is None:
            logger.warning(f'Instance is None for {term}')
        else:
            curr_vector = curr_predictor.predict_instance(instance)['vector']

            if len(curr_vector) == 1:
                curr_vector = curr_vector[0]

            term_to_vec[term] = curr_vector

    return term_to_vec

