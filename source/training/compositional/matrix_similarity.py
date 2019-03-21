import torch

from overrides import overrides
from torch.nn.parameter import Parameter

from allennlp.nn import Activation, util
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


@SimilarityFunction.register("matrix_composition")
class MatrixSimilarity(SimilarityFunction):
    """
    This similarity function computes f(xy) = tanh(W * [x ; y])
    for a learned matrix W and the given vectors x, y.
    ----------
    Parameters
    ----------
    input_dim : ``int``
        The dimension of the vectors.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    """
    def __init__(self,
                 input_dim: int,
                 activation : str = 'tanh') -> None:
        super(MatrixSimilarity, self).__init__()
        self._combination = "x,y"
        combined_dim = util.get_combined_dim(self._combination, [input_dim, input_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._activation = Activation.by_name(activation)()
        self.reset_parameters()

    def reset_parameters(self):
        self._weight_vector.data.uniform_(-1, 1)

    @overrides
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # If more than two words, take the average of each constituent
        if x.size()[1] > 1:
            x = torch.mean(x, dim=1).unsqueeze(1)

        if y.size()[1] > 1:
            y = torch.mean(x, dim=1).unsqueeze(1)

        combined_tensors = util.combine_tensors(self._combination, [x, y])
        dot_product = torch.matmul(combined_tensors, self._weight_vector)
        return self._activation(dot_product)
