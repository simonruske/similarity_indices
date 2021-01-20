import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from similarity import matching_matrix, adjusted_rand, TPQ_linkages
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage
from time import perf_counter

class TestMatchingMatrix(unittest.TestCase):
  
  def test_init_correct_defaults(self):
  
    # Arrange / Act
    m = matching_matrix(4)
    
    # Assert
    expected_rows = { 0 : {0:1}, 1 : {1:1}, 2 : {2:1}, 3:{3:1} }
    self.assertDictEqual(expected_rows, m.rows)
    
    expected_columns = { 0 : {0:1}, 1 : {1:1}, 2 : {2:1}, 3:{3:1} }
    self.assertDictEqual(expected_columns, m.columns)
    
    self.assertDictEqual({0:1, 1:1, 2:1, 3:1}, m.rtot)
    self.assertDictEqual({0:1, 1:1, 2:1, 3:1}, m.ctot)
    self.assertEqual(4, m.n)
    
    self.assertEqual(0, m.T)
    self.assertEqual(0, m.P)
    self.assertEqual(0, m.Q)
    
    self.assertDictEqual({}, m.sizes_A)
    self.assertDictEqual({}, m.update_A)
    self.assertDictEqual({}, m.sizes_B)
    self.assertDictEqual({}, m.update_B)
  
  def test_relabel_A_clusters_each_with_one_point(self):
  
    # set-up
    m = matching_matrix(10)
    
    i_1, i_2 = m.relabel_A(0, 1, 0)
    
    self.assertEqual(0, i_1)
    self.assertEqual(1, i_2)
    
    
    # there are 10 clusters 0-9 before the merge_cols
    # after the merge we create cluster 10 which should map to 1
    # in the row dictionary as we merge 0 into 1 arbitrarily taking the second
    # cluster to be the largest
    self.assertDictEqual({10:1}, m.update_A)
    
  def test_relable_A_clusters_first_with_2_second_with_1(self):
    
    # set-up
    m = matching_matrix(10)
    
    # merge 0 and 1 to create cluster 10 in step 0 and store it in column 1
    m.merge_rows(0, 1, 0)           
    
    # merge 10 (stored in column 1) and 2 to create cluster 11 in step 1 and continue to store in column 1
    i_1, i_2 = m.relabel_A(10, 2, 1) 
       
    self.assertEqual(2, i_1) # cluster 2 is the smallest
    self.assertEqual(1, i_2) # cluster 10 is the largest and is stored in column 1 
    
    self.assertDictEqual({11:1}, m.update_A)
  
  def test_worked_example(self):

    #================
    #-------A--------
    #================
    #  i1, i2 
              
    #[  3,  4 ]
    #[  1,  5 ]
    #[  0,  2 ]
    #[  6,  7 ]

    #================
    #-------B--------
    #================
    #  j1, j2, j_size

    #[  3,  4,  2]
    #[  0,  2,  2]
    #[  1,  5,  3]
    #[  6,  7,  5]

    # Arrange
    m = matching_matrix(5)
   
    ''' Merge 0 in A
        0, 1, 2, 3, 4
    0 [ 1, 0, 0, 0, 0] 1
    1 [ 0, 1, 0, 0, 0] 1
    2 [ 0, 0, 1, 0, 0] 1
    5 [ 0, 0, 0, 1, 1] 2
      1  1  1  1  1
    
    '''
    m.merge_rows(i_1 = 3, i_2 = 4, k = 0)
    
    self.assertEqual(0, m.T)
    self.assertEqual(1, m.P)
    self.assertEqual(0, m.Q)
    
    ''' Merge 0 in B
      0, 1, 2, 5
    0 [ 1, 0, 0, 0 ] 1
    1 [ 0, 1, 0, 0 ] 1
    2 [ 0, 0, 1, 0 ] 1
    5 [ 0, 0, 0, 2 ] 2
      1, 1, 1, 2
    '''
    
    m.merge_cols(j_1 = 3, j_2 = 4, k = 0)
    
    self.assertEqual(1, m.T)
    self.assertEqual(1, m.P)
    self.assertEqual(1, m.Q)
    
    ''' Merge 1 in A
        0, 1, 2, 5
    0 [ 1, 0, 0, 0 ] 1
    2 [ 0, 0, 1, 0 ] 1
    6 [ 0, 1, 0, 2 ] 3
        1, 1, 1, 2 
    '''
    m.merge_rows(i_1 = 1, i_2 = 5, k = 1)
    
    self.assertEqual(1, m.T)
    self.assertEqual(3, m.P)
    self.assertEqual(1, m.Q)
    
    ''' Merge 1 in B
        6  1  5 
    0 [ 1, 0, 0 ] 1
    2 [ 1, 0, 0 ] 1
    6 [ 0, 1, 2 ] 3
        2, 1, 2
    '''
    m.merge_cols(j_1 = 0, j_2 = 2, k = 1)
    
    self.assertEqual(1, m.T)
    self.assertEqual(3, m.P)
    self.assertEqual(2, m.Q)
    
    ''' Merge 2 in A
        6  1  5 
    7 [ 2, 0, 0 ] 2
    6 [ 0, 1, 2 ] 3
        2, 1, 2
    '''
    m.merge_rows(i_1 = 0, i_2 = 2, k = 2)
    
    self.assertEqual(2, m.T)
    self.assertEqual(4, m.P)
    self.assertEqual(2, m.Q)
    
    ''' Merge 2 in B
        6  7 
    7 [ 2, 0 ] 2
    6 [ 0, 3 ] 3
        2, 3
    '''
    m.merge_cols(j_1 = 1, j_2 = 5, k = 2)
    
    self.assertEqual(4, m.T)
    self.assertEqual(4, m.P)
    self.assertEqual(4, m.Q)
    
    
    ''' Merge 3 in A
        6  7 
    8 [ 2, 3 ] 5
        2, 3
    '''
    
    m.merge_rows(i_1 = 6, i_2 = 7, k = 3)
    
    self.assertEqual(4 , m.T)
    self.assertEqual(10, m.P)
    self.assertEqual(4 , m.Q)
   
    ''' Merge 3 in B
        8
    8 [ 5 ] 5
        5
    '''
    m.merge_cols(j_1 = 6, j_2 = 7, k = 3)
    
    self.assertEqual(10, m.T)
    self.assertEqual(10, m.P)
    self.assertEqual(10, m.Q)
    
class TestAdjustedRand(unittest.TestCase):

  def test_compare_example_with_sklearn(self):
  
    # Arrange
    A = np.array(
      [[ 1.        ,  6.        ,  0.60,  2.        ],
       [ 8.        ,  9.        ,  0.85,  2.        ],
       [ 4.        ,  5.        ,  1.50,  2.        ],
       [10.        , 12.        ,  1.56,  4.        ],
       [ 7.        , 13.        ,  2.42,  5.        ],
       [ 3.        , 11.        ,  2.60,  3.        ],
       [14.        , 15.        ,  2.63,  8.        ],
       [ 2.        , 16.        ,  4.14,  9.        ],
       [ 0.        , 17.        ,  5.14, 10.        ]]
      )
      
    B = np.array(
      [[ 1.        ,  6.        ,  0.60,  2.        ],
       [ 8.        ,  9.        ,  0.85,  2.        ],
       [ 4.        ,  5.        ,  1.50,  2.        ],
       [10.        , 12.        ,  2.21,  4.        ],
       [ 3.        , 11.        ,  3.00,  3.        ],
       [ 7.        , 13.        ,  3.07,  5.        ],
       [ 2.        , 15.        ,  4.73,  6.        ],
       [ 0.        , 14.        ,  4.76,  4.        ],
       [16.        , 17.        ,  7.68, 10.        ]]
      )
      
    # Act
    T, P, Q = TPQ_linkages(A, B)
    ar_similarity = adjusted_rand(T, P, Q, 10)
    
    ar_sklearn = []
    for i in range(9, 1, -1):
      fcluster_a = fcluster(A, i, 'maxclust')
      fcluster_b = fcluster(B, i, 'maxclust')
      ar = adjusted_rand_score(fcluster_a, fcluster_b)
      
      ar_sklearn.append(ar)
          
    # Assert
    assert_almost_equal(ar_similarity, ar_sklearn)
    
  def test_performance(self):

    # Arrange
    n = 100
    np.random.seed(seed = 8455624)
    x = np.random.normal(n, 2, (n, 2))
    A = linkage(x, 'centroid')
    B = linkage(x, 'ward')
    
    # Act
    
    similarity_times = []
    sklearn_times = []
    fcluster_times = []
    
    for repitition in range(100):
    
      start = perf_counter()
    
      T, P, Q = TPQ_linkages(A, B)
      ar_similarity = adjusted_rand(T, P, Q, n)
      
      end = perf_counter()
      
      similarity_times.append(end-start)
      
      ar_sklearn = []
      
      sklearn_time = 0
      fcluster_time = 0
      
      excluded_results = 0
      for i in range(n - 1, 1, -1):
      
        start = perf_counter()
        
        fcluster_a = fcluster(A, i, 'maxclust')
        fcluster_b = fcluster(B, i, 'maxclust')
        
        end = perf_counter()
        
        fcluster_time += (end - start)
        
        start = perf_counter()
        
        ar = adjusted_rand_score(fcluster_a, fcluster_b)
        
        end = perf_counter()
        
        sklearn_time += (end - start)
        
        # fcluster takes maxclust rather than an exact number of clusters 
        # most of the time it will create exactly maxclust, but for the occassions 
        # that it doesn't the results are are not comparable so ignore them
        if (len(np.unique(fcluster_a)) != i) or (len(np.unique(fcluster_b)) != i):
          excluded_results += 1
          ar_sklearn.append(ar_similarity[len(ar_sklearn)])
          
        else:
          ar_sklearn.append(ar)
       
      sklearn_times.append(sklearn_time)
      fcluster_times.append(fcluster_time)
      
      ar_sklearn = np.array(ar_sklearn)
       
      idx = ar_sklearn != np.nan
      
      # Assert
      self.assertEqual(len(ar_sklearn), len(ar_similarity))
      assert_almost_equal(ar_similarity, ar_sklearn)
      self.assertEqual(4, excluded_results) # double-check that we haven't excluded everything
    
    print("\nSimilarity average time: ", np.average(similarity_times))
    print("\nSklearn average time: ", np.average(sklearn_times))
    print("\nFCluster average time: ", np.average(fcluster_times))
    
  
  
if __name__ == '__main__':
  unittest.main()