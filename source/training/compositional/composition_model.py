import torch

from typing import Dict
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
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
    composition: ``SimilarityFunction`` to compute the composition of x and y.

    Choose between:
        ``AddSimilarity'': to compute f(xy) = a * x + b * y, where a and b are scalars.
        ``FullAddSimilarity'': to compute f(xy) = A * x + B * y, where A and B are matrices.
        ``MatrixSimilarity'' to compute f(xy) = tanh(W * [x ; y]),
        where g is a non-linearity and W is a matrix.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 composition_function: SimilarityFunction) -> None:
        super(CompositionModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.composition_function = composition_function
        self.loss = torch.nn.MSELoss()
        self.metrics = {}

    @overrides
    def forward(self,  # type: ignore
                nc: Dict[str, torch.LongTensor],
                w1: Dict[str, torch.LongTensor],
                w2: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
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

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # Embed
        nc_obs = self.text_field_embedder(nc)
        w1_emb = self.text_field_embedder(w1)
        w2_emb = self.text_field_embedder(w2)

        # Compose
        nc_cmp = self.composition_function(w1_emb, w2_emb)

        # Compute the loss
        output_dict = {'loss': self.loss(nc_obs, nc_cmp),
                       'vector': nc_cmp}
        return output_dict
