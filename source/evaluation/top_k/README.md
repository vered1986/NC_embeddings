## Top k

Computes the top k similar neighbours for each 
noun compound in the test set. 

```
usage: top_k.py [-h] [--is_compositional] emb_file embedding_dim nc_file

positional arguments:
  emb_file            The gzipped text embedding file
  embedding_dim       The embedding dimension
  nc_file             The test/val set file

optional arguments:
  -h, --help          show this help message and exit
  --is_compositional  Whether the embeddings are from a compositional/paraphrase-based model
```

To run all:

```
bash compute_top_k.sh
```