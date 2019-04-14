import torch

from overrides import overrides
from typing import Dict, Optional

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction



@Model.register("composition_model")
class CompositionModel(Model):
    """
    This ``Model`` composes the constituents of a noun compound.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    composition: ``SimilarityFunction`` to compute the composition of x and y, optional

    Choose between:
        ``AddSimilarity'': to compute f(xy) = a * x + b * y, where a and b are scalars.
        ``FullAddSimilarity'': to compute f(xy) = A * x + B * y, where A and B are matrices.
        ``MatrixSimilarity'' to compute f(xy) = tanh(W * [x ; y]),
        where g is a non-linearity and W is a matrix.

    encoder : ``Seq2VecEncoder``, optional
        The RNN encoder that returns a vector given a list of vectors.

    One of ``composition`` or ``encoder`` must be given.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 composition_function: Optional[SimilarityFunction] = None,
                 encoder: Optional[Seq2VecEncoder] = None) -> None:
        super(CompositionModel, self).__init__(vocab)

        if (composition_function is None and encoder is None) or \
                (composition_function is not None and encoder is not None):
            raise ValueError('Exactly one of composition or encoder must be given')

        self.text_field_embedder = text_field_embedder
        self.composition_function = composition_function
        self.encoder = encoder
        self.loss = torch.nn.MSELoss()
        self.metrics = {}

    @overrides
    def forward(self,  # type: ignore
                nc: Dict[str, torch.LongTensor],
                w1: Dict[str, torch.LongTensor],
                w2: Dict[str, torch.LongTensor],
                nc_seq: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        nc : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        w1 : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        w2 : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        nc_seq : Dict[str, Variable], required
            The output of ``TextField.as_array()``.

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # Embed
        nc_obs = self.text_field_embedder(nc)

        # Compose
        if self.composition_function:
            w1_emb = self.text_field_embedder(w1)
            w2_emb = self.text_field_embedder(w2)
            nc_cmp = self.composition_function(w1_emb, w2_emb).squeeze()
        else:
            nc_emb = self.text_field_embedder(nc_seq)
            nc_mask = util.get_text_field_mask(nc_seq)
            nc_cmp = self.encoder(nc_emb, nc_mask)

        # Compute the loss
        output_dict = {'loss': self.loss(nc_obs, nc_cmp),
                       'vector': nc_cmp}
        return output_dict
