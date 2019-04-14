import json
import random
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
class NCParaphraseDatasetReader(DatasetReader):
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
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as f_in:
            all_examples = [json.loads(line.strip()) for line in f_in]

        logger.info("Generating negative paraphrases...")
        constituents = [example['compound'].split(' ') for example in all_examples]
        all_paraphrases = [p.replace(constituents[i][0], '[w1]').replace(constituents[i][1], '[w2]')
                           for i, example in enumerate(all_examples)
                           for p in example['paraphrases']]

        for example in all_examples:
            nc = example['compound']
            w1, w2 = nc.split()
            negative_sample = [p.replace('[w1]', w1).replace('[w2]', w2)
                               for p in random.sample(all_paraphrases, 10)]

            neg_paraphrase = [p for p in negative_sample if p not in set(example['paraphrases'])][0]

            for paraphrase in example['paraphrases']:
                instance = self.text_to_instance(nc, paraphrase, neg_paraphrase)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, nc: str, paraphrase: str = None, neg_paraphrase: str = None) -> Instance:
        tokenized_nc = self._tokenizer.tokenize(nc)
        nc_field = TextField(tokenized_nc, self._token_indexers)

        # Remove non-binary NCs to make it comparable to the other composition functions
        if nc_field.sequence_length() != 2:
            return None

        fields = {'nc': nc_field}

        if paraphrase is not None:
            tokenized_paraphrase = self._tokenizer.tokenize(paraphrase)
            paraphrase_field = TextField(tokenized_paraphrase, self._token_indexers)
            fields['paraphrase'] = paraphrase_field

            tokenized_neg_paraphrase = self._tokenizer.tokenize(neg_paraphrase)
            neg_paraphrase_field = TextField(tokenized_neg_paraphrase, self._token_indexers)
            fields['neg_paraphrase'] = neg_paraphrase_field

        return Instance(fields)
