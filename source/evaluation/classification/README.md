## Noun-Compound Relation Classification

In this task, noun-compounds are annotated to a pre-defined set of relations, and the model has to predict the correct 
relation between the constituents of an unobserved compound. For example, `olive oil` may belong to the `SOURCE` relation 
while `morning meeting` belongs to the `TIME` relation. 

We train models whose input is the noun compound representation. 

```
usage: classifier.py [-h] embedding_path

positional arguments:
  embedding_path    zipped textual word embedding file containing the 
                    embeddings of the noun compounds. 
                    For compositional embeddings, first run 
                    training/compositional/compute_vectors.py

optional arguments:
  -h, --help            show this help message and exit
```
