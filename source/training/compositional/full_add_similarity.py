import torch

from overrides import overrides
from torch.nn.parameter import Parameter

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


@SimilarityFunction.register("full_add_composition")
class FullAddSimilarity(SimilarityFunction):
    """
    This similarity function computes f(xy) = A * x + B * y
    for learned matrices A, B and the given vectors x, y.
    ----------
    Parameters
    ----------
    input_dim : ``int``
        The dimension of the vectors.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    """
    def __init__(self,
                 input_dim: int) -> None:
        super(FullAddSimilarity, self).__init__()
        self._A = Parameter(torch.Tensor(input_dim, input_dim))
        self._B = Parameter(torch.Tensor(input_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self._A.data.uniform_(-1, 1)
        self._B.data.uniform_(-1, 1)

    @overrides
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self._A) + torch.matmul(y, self._B)
