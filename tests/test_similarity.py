import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from similarity import TPQ_linkages, adjusted_rand
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
 
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
  
if __name__ == '__main__':
  unittest.main()