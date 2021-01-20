import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from fastcluster import linkage
from time import perf_counter
from similarity import TPQ_linkages, adjusted_rand
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

class Performance(unittest.TestCase):

  def test_adjusted_rand_performance(self):

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