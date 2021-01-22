# Similarity metrics for hierarchical clusterings

Calculation of similarity indices such as the adjusted rand score to compare two hierarchical clusterings

# Getting started

As an example, we can calculate the adjusted rand score for each level of the two hierarchical clusterings represented by linkage
matrices `A` and `B` by doing the following

```python
  from similarity import similarity_metrics
  
  metrics = similarity_metrics(A, B)
  ar_similarity = metrics.adjusted_rand()
```
## Current Priorities
* Add better documentation
* Move the experimental methods into the main file after testing the supporting matching matrices

## Coverage

| Module                               | statements | missing | excluded | coverage |
|--------------------------------------|-----------:|--------:|---------:|---------:|
| matching_matrices\matching_matrix.py |         76 |       0 |        0 |     100% |
| similarity.py                        |         47 |       2 |        0 |      96% |