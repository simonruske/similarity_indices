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
# Current Priorities
* Improve documentation
* Move the experimental methods into the main file after testing the supporting matching matrices

# Coverage

| Module                               | statements | missing | excluded | coverage |
|--------------------------------------|-----------:|--------:|---------:|---------:|
| matching_matrices\matching_matrix.py |         76 |       0 |        0 |     100% |
| similarity.py                        |         47 |       2 |        0 |      96% |

# Implementation Details

## Matching Matrix

When comparing two linkage matrices `A` and `B` the `matching_matrix` class is used to store the matching matrix as we merge clusters in the hierarchies. Initially it starts with an identity matrix and then `merge_rows` and `merge_columns` are used to condense the matrix as we step through the hierarchies. As we merge the columns and rows we keep track of serveral quantities, which we will consider below.

First let's examine the data structure used to store the matrix. Here I take inspiration from a dictionary of keys (DoK) based sparse matrix. Firstly we should consider that the matching matrix has dimensions ranging from between `N` x `N` and `1` x `1` but contains at most `N` elements and therefore a sparse structure is appropriate. The DoK structure however does not lend itself well to quick retrival for rows and columns so it is necessary to make suitable modifications for our use case. Instead of storing a map between row and column indices to value we store the following

* A map from row index to row dictionary, where the row dictionary is it itself a map between column indices and values.
* A map from the column index to column dictionary, where the column dictionary is itself a map between the row indices and values.

One of the drawbacks of this approach is that the matrix is stored twice. But since there are only at most `N` values in the matrix this keeps the memory consumption to the same order as the original matrices `A` and `B`.

The remaining objects that we store are 
* a map between row/column indices and row/column totals (used for updating `P` and `Q`),
* the current value of `T`, `P` and `Q`,
* update dictionaries which we detail in the following section.

## Step I : Relabelling

```python
  if i_1 >= self.n: 
    i_1 = self.update_A.pop(i_1) 

  if i_2 >= self.n: 
    i_2 = self.update_A.pop(i_2) 

  if (self.rtot[i_1] > self.rtot[i_2]):
    i_1, i_2 = i_2, i_1
        
  self.update_A[k + self.n] = i_2

  return i_1, i_2
```

This is a non-essential step that I added in order to increase the performance.

To understand what is happening in this step and why, let's examine the process for the first hierarchy `A`. At every stage of the hierarchy a merge takes place between two clusters `i_1` and `i_2`. When we are converting the `k`th matching matrix to the `(k+1)`th matching matrix, we need to reduce the number of rows by `1` by combining rows `i_1` and `i_2` into a single row. `Fastcluster` uses index `n + k` as the label for the newly formed cluster so we create a map between $n+k$ to the index of the cluster with the largest number of points. This map is stored in the `update_a` dictionary. We keep this record so when we need to retrieve the row in subsequent merges we know where to find it. We merge the cluster with the fewest points into the cluster the the largest number of points to decrease the number of insertions/merges we need to do. For example, if cluster `i_1` has `20` points and cluster `i_2` has `2` points it is far more efficient to make two insertions/combinations into the dictionary corresponding to cluster `i_1` than it is to make `20` insertions into the dictionary corresponding to cluster `i_2`.

### Step II : Update the value for P

Without loss of generality, we discuss only rows and `P` and not `Q`.

```python
  rtot1, rtot2 = self.rtot.pop(i_1), self.rtot[i_2]
  self.rtot[i_2] = rtot1 + rtot2
  self.P += rtot1 * rtot2
```

Remember, that `P` is the total number of points placed into the same cluster for the current stage of hierarchy A. When we merge the two clusters `i_1` and `i_2`, we can calculate the change in `P` by subtracting the pairs of point in the original clusters `i_1` and `i_2` and adding the pairs of points in the newly formed cluster i.e.

```python
self.P += (rtot1 + rtot2) * (rtot1 + rtot2 + 1) / 2  - rtot1 * (rtot1 - 1) / 2 - rtot2 * (rtot2 - 1) / 2 
```

which simplifies to 
```python
self.P += rtot1 * rtot2
```

### Step III : Merge the two clusters
As was the case in Step II, we consider only the row dictionary without loss of generality. 

First we retrieve the rows and store them in `r1` and `r2`. Since we are merging `r1` into `r2` we pop row `i_1` and keep row `i_2`.

```python
r1, r2 = self.rows.pop(i_1), self.rows[i_2]
```

Next we loop over all of the elements in the smallest cluster (the relabelling procedure ensures `i_1` is the smallest). If an entry for the column index is not already present in `r2`, then we have merged with a non-zero entry. This results in `T` not change as no new pairs of points have been created. It is therefore only necessary to update r2 and the column storage with the value

Conversely, if we find there is an entry in `r2` with the same column index as the element in `r1` then we have increased the number of pairs of points placed into the same cluster for both `A` and `B` and therefore we need to update `T`. The update formula is the same as the update for `P` except it is performed on entries in the matrix rather than the row totals.

```python
for elem in r1: 

    if elem in r2: 
        value_1 = self.columns[elem].pop(i_1)
        value_2 = self.columns[elem][i_2]
        value_new = value_1 + value_2
        r2[elem] = self.columns[elem][i_2] = value_new
             self.T += value_1 * value_2

    else:
        r2[elem] = self.columns[elem][i_2] = self.columns[elem].pop(i_1)

```



