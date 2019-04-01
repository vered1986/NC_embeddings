import json
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


@DatasetReader.register("nc_paraphrases_data_reader")
class NCDatasetReader(DatasetReader):
    """
    Reads a jsonl file containing a list of noun compounds, each with paraphrases
    obtained from back-translation,
    and creates a dataset for the training of compositional method.

    Expected format for each input line: { 'compound' : w1_w2 , 'paraphrases' : list }

    The output of ``read`` is a list of ``Instance`` s with the fields:
        nc: ``TextField``
        paraphrase: ``TextField``

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
                example = json.loads(line.strip())

                for paraphrase in example['paraphrases']:
                    yield self.text_to_instance(example['compound'], paraphrase)

    @overrides
    def text_to_instance(self, nc: str, paraphrase: str) -> Instance:
        tokenized_nc = self._tokenizer.tokenize(nc)
        nc_field = TextField(tokenized_nc, self._token_indexers)

        tokenized_paraphrase = self._tokenizer.tokenize(paraphrase)
        paraphrase_field = TextField(tokenized_paraphrase, self._token_indexers)

        fields = {'nc': nc_field, 'paraphrase': paraphrase_field}
        return Instance(fields)
