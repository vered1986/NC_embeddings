# Compositional Embeddings

Learning vectors by composing the embeddings of the constituent words.

The compositional methods rely on the distributional vectors 
of both the noun compounds and their constituents. 
They are obtained by training the 
[distributional embeddings](../distributional/README.md), 
and then learning the composition function.

We implemented 3 composition functions:

- **Add** [(Mitchell and Lapata, 2009)](http://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2010.01106.x/pdf) - f(xy) = alpha * x + beta * y for scalars alpha, beta.
- **FullAdd** [(Zanzotto et al., 2010)](http://www.aclweb.org/anthology/C10-1142) - f(xy) = A * x + B * y for matrices A, B.
- **Matrix** [(Socher et al., 2012)](http://aclweb.org/anthology/D12-1110) - f(xy) = g(W * [x ; y] + b) for non-linearity g and matrix W.

To train:

```
bash train_composition_models.sh
```
