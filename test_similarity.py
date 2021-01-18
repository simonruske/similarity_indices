import unittest
from similarity import matching_matrix

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
    
  
if __name__ == '__main__':
  unittest.main()