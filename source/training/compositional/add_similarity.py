import math
import torch

from overrides import overrides
from torch.nn.parameter import Parameter

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


@SimilarityFunction.register("add_composition")
class AddSimilarity(SimilarityFunction):
    """
    This similarity function computes f(xy) = a * x + b * y
    for learned scalars a, b and the given vectors x, y.
    """
    def __init__(self) -> None:
        super(AddSimilarity, self).__init__()
        self._a = Parameter(torch.Tensor(1,))
        self._b = Parameter(torch.Tensor(1,))
        self.reset_parameters()

    def reset_parameters(self):
        self._a.data.uniform_(-1, 1)
        self._b.data.uniform_(-1, 1)

    @overrides
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self._a) + torch.mul(y, self._b)
