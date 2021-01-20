import numpy as np

from matching_matrices.matching_matrix import matching_matrix
from matching_matrices.matching_matrix_proportion import matching_matrix_proportion
from matching_matrices.matching_matrix_pairs import matching_matrix_pairs

from scipy.cluster.hierarchy import is_valid_linkage
from collections import Counter

def proportion(A, B): 

    n = len(A) + 1 
    prop = np.zeros(n+1)
    m = matching_matrix_proportion(B, n)

    for k, rows_A in enumerate(A): 
        m.merge_rows(rows_A[0], rows_A[1], rows_A[3], k)
        maximums = m.maximums
        where = m.where 
        total = 0 
        for index in Counter(where).keys():
           total += max(maximums[where == index])
           prop[len(m.rows)] = float(total) / m.n

    return(prop)

def TPQ_linkages(A, B):

    """
    Calculates statistics on two hierarchical clusterings of the same set of objects 
    which can be used to calculate indices to compare two hierarchical clusterings

    Parameters
    ----------
    A : ndarray
        A :math:`(n-1)` by 4 matrix encoding the linkage
        (hierarchical clustering).  See ``linkage`` documentation
        for more information on its form.
    B : A second :math:`(n-1)` by 4 matrix encoding the linkage
        (hierarchical clustering).
    
    Returns
    -------
    T : ndarray
        A vector of size :math:'n-1' where the ``k``'th element
        contains the number of pairs of objects placed into the 
        same cluster for both hierarchical clusterings after the 
        clusters in the ``k``'th row have been merged.  
    P : ndarray 
        A vector of size :math:'n-1' where the ``k``'th element 
        contains the number of pairs of objects placed into the
        same cluster after the clusters in the ``k``th row 
        for the hierarchical clustering A only.
    Q : ndarray 
        A vector of size :math:'n-1' where the ``k``'th element
        contains the number of pairs of objects placed into the
        same cluster after the clusters in the ``k``th row 
        for the hierarchical clustering B only.
    """
 
    # Convert to array if not already.
    A = np.array(A, 'double')
    B = np.array(B, 'double')
    
    # Performs checks
    is_valid_linkage(A, throw=True)
    is_valid_linkage(B, throw=True)
    
    n = len(A) + 1
    n2 = len(B) + 1
    
    if n != n2: 
        raise ValueError("The hierarchical clusterings must be of the same size")

    T = np.zeros(n-1)
    P = np.zeros(n-1)
    Q = np.zeros(n-1)
        
    # Creates a new matching matrix (identity of size n)
    m = matching_matrix(n)
    
    # Merges the required clusters as specified by the input files 
    for k, (rows_A, rows_B) in enumerate(zip(A, B)):
        T[k], P[k], Q[k] = m.merge(rows_A[0], rows_A[1], rows_B[0], rows_B[1], k)
        
    #Outputs the values of T, P and Q. 
    return T[:-1], P[:-1], Q[:-1] 
        
def TPQ_known(A, B, num = None):

    '''
    TODO : Docstring
    ''' 

    n = len(A) + 1
    T = np.zeros(n-1)
    P = np.zeros(n-1)
    Q = np.zeros(n-1)
         
    # Creates a new matching matrix (identity of size n)
    m = matching_matrix_pairs(B, n)
    
    # Merges the required clusters as specified by the input files 
    for k, rows_A in enumerate(A):
        m.merge_rows(rows_A[0], rows_A[1], rows_A[3], k)
        T[k], P[k], Q[k] = m.T, m.P, m.Q
 
    return T[:-1], P[:-1], Q[:-1] 
    
def similarity(T, P, Q, index, n = None):

    """
    Calculates an index to determine how similar the two 
    hierarchical clusterings A and B are. 
    
    The following indices are available. 
  
      * index='AR', Adjusted
        .. math:: 
           AR_k = \\frac{2T_k-\\frac{P_k Q_k}{n(n-1)}}{P_k+Q_k - \\frac{P_k+Q_k}{n(n-1)}}

      * index='R, Rand
        .. math::
           R_k = \\frac{n(n-1)-2(2 T_k - P_k - Q_k)}{n(n-1)}
      
      * index='B', Fowlkes and Mallows index 
        .. math:: 
           B_k = \\frac{T_k}{\\sqrt{P_k Q_k}}
     
      * index='S', Morlini and Zani's S index  
        .. math:: 
           S = \\frac{2 \\sum_{k=2}^{n-1} T_k}{\\sum_{k=2}^{n-1} P_k + \\sum_{k=2}^{n-1} Q_k}

      * index='SK', Morlini and Zani's SK index
        .. math::
           S_k = \\frac{2 T_k}{P_k+Q_k} 

    With the exception of the Adjusted Rand each of the indices are defined on 
    the interval :math:`[0,1]` where values close to 1 indicate strong 
    similarity and values close to 0 indicate lack of similairity. The same is 
    true for the Adjusted Rand index except it is possible that the index can 
    take on values in the interval :math:`[-1,0)` when the value of the Rand 
    index is less than it's expected value. 
 
    Parameters
    ----------
   
    T : ndarray 
        The output T from the function TPQ.
  
    P : ndarray 
        The output P from the function TPQ. 
  
    Q : ndarray 
        The output Q from the function TPQ.
   
    index : string 
        Either 'AR', 'R', 'B' or 'S' indicating which index you would 
        like to use.
    
    Returns
    -------
  
    result : ndarray or float
   
        With the exception of the S-index the result with be 
        a vector of size :math:'n-1' where the :math:``k``'th element
        contains an index indicating how similar the clusterings A 
        and B are after the :math: . For the S-index the result will 
        be a single index.  

    """ 
    
    if index == 'S':
        return s_index(T, P, Q)
      
    elif index == 'SK':
        return sk_index(T, P, Q)
      
    elif index == 'R':
        if n == None: 
            n = len(T) + 2
        return rand(T, P, Q, n)
    
    elif index == 'B': 
        return fowlkes_mallows(T, P, Q)
    
    elif index == 'AR': 
        if n == None: 
            n = len(T) + 2
        return adjusted_rand(T, P, Q, n)

    elif index == 'M':
        return mirkin(T, P, Q)

    elif index == 'J': 
        return jaccard(T, P, Q)

def rand(T, P, Q, n):
    N = n * (n-1) // 2 
    check_TPQ(T, P, Q)
    return (N - P - Q + 2 * T) / N 

def adjusted_rand(T, P, Q, n):
    N = n * (n-1) // 2
    check_TPQ(T, P, Q)
    return  2 * (N * T - P * Q) / (N * (P + Q) - 2 * P * Q)

def fowlkes_mallows(T, P, Q):
    check_TPQ(T, P, Q)
    return T / np.sqrt(P * Q)

def s_index(T, P, Q):
    check_TPQ(T, P, Q)
    return 2 * sum(T[:-1]) / ( sum(P[:-1]) + sum(Q[:-1]) )

def sk_index(T, P, Q):
    check_TPQ(T, P, Q)
    return 2 * T / (P + Q)

def mirkin(T, P, Q): 
    check_TPQ(T, P, Q)
    return 2 * (P + Q - 2*T)

def adjusted_mirkin(T, P, Q, n): 
    check_TPQ(T, P, Q) 
    return 2 * (P + Q - 2*T) / n ** 2

def jaccard(T, P, Q): 
    check_TPQ(T, P, Q) 
    return T / (P + Q - T)

def check_TPQ(T,P,Q): 

    (n, n2, n3) = (len(T), len(P), len(Q))
    
    if (n != n2) or (n != n3) or (n2 != n3): 
        raise InputError("T, P and Q must be vectors of the same length")

