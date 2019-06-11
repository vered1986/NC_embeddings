# Paraphrase-based Embeddings

Learning vectors by minimizing their distance to the vectors of their paraphrases.

The paraphrase based methods rely on the distributional vectors 
of the noun compound constituents. 
They are obtained by training the 
[distributional embeddings](../distributional/README.md), 
and then learning the paraphrase-based function.

We implemented 2 paraphrase-based functions:

- **Co-occurrence** [(Shwartz and Dagan, 2018)](https://aclweb.org/anthology/P18-1111) - in which the paraphrases are co-occurrences of the constituent words in the corpus (i.e. "oil for baby" is a paraphrase of "baby oil").
- **Backtranslation** [(Wieting et al., 2016)](https://arxiv.org/pdf/1511.08198.pdf) - in which the paraphrases are translations of the noun compounds to a foreign language and back to English (e.g. "ground floor" for "street level").

To train:

```
bash train_paraphrase_based.sh
```

To predict the embeddings of the test set:

```
bash predict_paraphrase_based.sh
```

