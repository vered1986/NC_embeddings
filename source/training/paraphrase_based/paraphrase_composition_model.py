import math
import torch

from overrides import overrides
from typing import Dict, Optional

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.common.checks import check_dimensions_match
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder



@Model.register("paraphrase_composition_model")
class CompositionModel(Model):
    """
    This ``Model`` composes the constituents of a noun compound via LSTM.

    Training is done by minimizing the distance between each noun compound and each of its paraphrases.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``, optional
        The RNN encoder that returns a vector given a list of vectors.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Optional[Seq2VecEncoder] = None) -> None:
        super(CompositionModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.metrics = {}

        self.encoder = encoder
        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text embedding dim", "encoder input dim")

    @overrides
    def forward(self,  # type: ignore
                nc: Dict[str, torch.LongTensor],
                paraphrase: Dict[str, torch.LongTensor] = None,
                neg_paraphrase: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        nc : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        paraphrase : Variable, optional (default = None)
            The output of ``TextField.as_array()`` for the paraphrase (given during training).
        neg_paraphrase: Variable, optional (default = None)
            The output of ``TextField.as_array()`` for a negative sampled paraphrase (given during training).

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # Embed and encode the noun compound
        nc_emb = self.text_field_embedder(nc)
        nc_mask = util.get_text_field_mask(nc)
        nc_enc = self.encoder(nc_emb, nc_mask)

        output_dict = {'vector': nc_enc}

        # Embed and encode the paraphrase
        if paraphrase is not None:
            paraphrase_emb = self.text_field_embedder(paraphrase)
            paraphrase_mask = util.get_text_field_mask(paraphrase)
            paraphrase_enc = self.encoder(paraphrase_emb, paraphrase_mask)

            neg_paraphrase_emb = self.text_field_embedder(neg_paraphrase)
            neg_paraphrase_mask = util.get_text_field_mask(neg_paraphrase)
            neg_paraphrase_enc = self.encoder(neg_paraphrase_emb, neg_paraphrase_mask)

            # Compute the loss:

            # Similarity to the paraphrase
            sim_p = (nc_enc * paraphrase_enc).sum(dim=-1)
            sim_p /= (math.sqrt((nc_enc ** 2).sum(dim=-1)) * math.sqrt((paraphrase_enc ** 2).sum(dim=-1)))

            # Similarity to a random sampled paraphrase
            sim_n = (nc_enc * neg_paraphrase_enc).sum(dim=-1)
            sim_n /= (math.sqrt((nc_enc ** 2).sum(dim=-1)) * math.sqrt((neg_paraphrase_enc ** 2).sum(dim=-1)))

            loss = self.margin - sim_p + sim_n
            loss = loss * (loss > 0)

            output_dict['loss'] = loss

        return output_dict
