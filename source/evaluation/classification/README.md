## Noun-Compound Relation Classification

In this task, noun-compounds are annotated to a pre-defined set of relations, and the model has to predict the correct 
relation between the constituents of an unobserved compound. For example, `olive oil` may belong to the `SOURCE` relation 
while `morning meeting` belongs to the `TIME` relation. 

We train models whose input is the noun compound representation. 

```
usage: classifier.py [-h] embedding_path dataset_prefix model_dir

positional arguments:
  embedding_path    zipped textual word embedding file containing the 
                    embeddings of the noun compounds. 
                    For compositional embeddings, first run 
                    training/compositional/compute_vectors.py
              
  dataset_prefix    path to the train/test/val/rel data
        
  model_dir         where to store the result

optional arguments:
  -h, --help            show this help message and exit
```

To run all:

```
bash train_all_classifiers.sh
```