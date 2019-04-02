import logging

from typing import Dict
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import LabelField, TextField, IndexField, SpanField

logger = logging.getLogger(__name__)


@DatasetReader.register("nc_data_reader")
class NCDatasetReader(DatasetReader):
    """
    Reads a text file containing a list of noun compounds,
    and creates a dataset for the training of compositional methods.

    Expected format for each input line: w1_w2

    The output of ``read`` is a list of ``Instance`` s with the fields:
        nc: ``TextField``
        w1: ``TextField``
        w2: ``TextField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                nc = line.strip()
                nc = nc.lower().replace('\t', '_')
                w1, w2 = nc.split('_')
                instance = self.text_to_instance(nc, w1, w2)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, nc: str, w1: str, w2: str) -> Instance:
        tokenized_nc = self._tokenizer.tokenize(nc)
        nc_field = TextField(tokenized_nc, self._token_indexers)

        # Remove non-binary NCs
        if nc_field.sequence_length() != 1:
            return None

        tokenized_w1 = self._tokenizer.tokenize(w1)
        w1_field = TextField(tokenized_w1, self._token_indexers)
        tokenized_w2 = self._tokenizer.tokenize(w2)
        w2_field = TextField(tokenized_w2, self._token_indexers)
        tokenized_nc_seq = self._tokenizer.tokenize(' '.join((w1, w2)))
        nc_seq_field = TextField(tokenized_nc_seq, self._token_indexers)

        fields = {'nc': nc_field, 'w1': w1_field, 'w2': w2_field, 'nc_seq': nc_seq_field}
        return Instance(fields)
