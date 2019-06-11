## Noun-Compound Property Classification

In this task, we have multiple datasets each corresponding to a single
property, e.g. "is round". Noun-compounds are annotated to 
whether they hold this property or not, 
and the model has to predict the correct 
answer of an unobserved compound. For example, `apple cake` is 
a positive instance for the "is round" task. 

We train models whose input is the noun compound representation. 

```
usage: classifier.py [-h] model_path dataset_prefix model_dir

positional arguments:
  model_path      word embeddings or composition model path
  dataset_prefix  path to the train/test/val/rel data
  model_dir       where to store the result

optional arguments:
  -h, --help      show this help message and exit
```

The data can be found under [data](data). 

To run all:

```
bash train_all_classifiers.sh
```