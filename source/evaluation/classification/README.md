## Noun-Compound Relation Classification

In this task, noun-compounds are annotated to a pre-defined set of relations, and the model has to predict the correct 
relation between the constituents of an unobserved compound. For example, `olive oil` may belong to the `SOURCE` relation 
while `morning meeting` belongs to the `TIME` relation. 

We train models whose input is the noun compound representation. 

```
usage: classifier.py [-h] in_model_path dataset_prefix out_model_dir

positional arguments:
  in_model_path   word embeddings or composition model path
  dataset_prefix  path to the train/test/val/rel data
  out_model_dir   where to store the result

optional arguments:
  -h, --help      show this help message and exit
```

We use the [Tratz (2011)](http://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/176191) 
dataset in several variants (fine or coarse-grained 
relation inventory, random or lexical data split), 
which can be found under [data](data). 

To run all:

```
bash train_all_classifiers.sh
```