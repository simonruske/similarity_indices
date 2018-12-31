import numpy as np
from scipy.cluster.hierarchy import is_valid_linkage
from collections import Counter

class matching_matrix():

    """        
    Class used to store a matching matrix. The matching matrix is used 
    to store information about two clusterings of a set of :math:n 
    objects. We call the first clustering :math:`A` and the second :
    math:`B`. Each row represents a cluster :math:`i` from the 
    clustering :math:`A`. Each column represents a cluster :math:`j` 
    from the clustering :math:`B`. Each element of the matching matrix 
    will then be the number of objects in the cluster :math:`i` that are 
    also in the cluster :math:`j`. 
  
    Initially for both clusterings we let each object be in a cluster on 
    it's own so the matching matrix is an identity matrix of size n. The
    merge function is then used to successively merge two clusters from 
    :math:`A` and :math:`B` until there is only one cluster left in each 
    and the matching matrix is :math:`[n]` 

    Parameters
    ----------
    n : integer
        An integer for the size of the initial matching matrix.  

    """

    def __init__(self, n, * args, **kwargs):        

        # The matrix itself 
        self.rows = {x : {x:1} for x in range(n)}
        self.columns = {x : {x:1} for x in range(n)}
        
        # Row and column totals 
        self.rtot = {x : 1 for x in range(n)}
        self.ctot = {x : 1 for x in range(n)}
        self.n = n
        
        # TPQ
        self.T = 0 
        self.P = 0 
        self.Q = 0 

        # Dictionaries used for the relabelling procedure   
        self.sizes_A = {}
        self.update_A = {}
        self.sizes_B = {}
        self.update_B = {}

    def merge_rows(self, i_1, i_2, isize, k): 
        
        """

        Parameters 
        ----------

        i_1 : int 
             The label of the first cluster to be merged in 
             hierarhical clustering A

        i_2 : int 
            The label of the second cluster to be merged in 
            hierarchical clustering A 

        isize: int 
            The size of the resulting cluster in the
            hierarchical clustering A once the merge 
            has taken place. 

        k: int 
            The number of merges that have already taken place 
            in the hierarchical clustering A. 

        """

        # Relabelling procedure for A. This is used so the new cluster 
        # can be created by adding the smallest old cluster onto the 
        # largest old cluster. This is to save substantial time in 
        # the case of some inputs.        
        if i_1 >= self.n: 
            i_1 = self.update_A.pop(i_1) 

        if i_2 >= self.n: 
            i_2 = self.update_A.pop(i_2) 

        if (self.rtot[i_1] >= self.rtot[i_2] and
            i_1 >= self.n and i_2 >= self.n): 
            i_1 , i_2 = i_2, i_1
        
        self.update_A[k + self.n] = i_2 

        # Updates the row totals and the value of P
        rtot1, rtot2 = self.rtot.pop(i_1), self.rtot[i_2]
        self.rtot[i_2] = rsum = rtot1 + rtot2
        self.P += sum([rsum**2, -rtot1**2, -rtot2**2]) // 2

        # Procedure used to merge the clusters in A, the variable 'st' 
        # is used to store the increase in T. The column version of 
        # the matrix is updated at the same time. We add the smallest 
        # row r1 to the largest row r2. 
        
        r1, r2, st = self.rows.pop(i_1), self.rows[i_2], 0
        
        # For every element in the smallest row, if it's not there then 
        # add it, if it is then add it on and update the value of T. In 
        # either case update the column dictionary with the change.  

        for elem in r1: 
          
            if elem not in r2:
                r2[elem]=self.columns[elem][i_2] = self.columns[elem].pop(i_1)
 
            else: 
                value_1 = self.columns[elem].pop(i_1)
                value_2 = self.columns[elem][i_2]
                value_new = value_1 + value_2
                r2[elem] = self.columns[elem][i_2] = value_new
                st += sum((value_new**2,-value_1**2, -value_2**2))

        self.T += st // 2
        self.rows[i_2] = r2


    def merge_cols(self, j_1, j_2, jsize, k):

        """

        Parameters 
        ----------

        j_1 : int 
             The label of the first cluster to be merged in 
             hierarhical clustering B

        j_2 : int 
            The label of the second cluster to be merged in 
            hierarchical clustering B 

        jsize: int 
            The size of the resulting cluster in the
            hierarchical clustering B once the merge 
            has taken place. 

        k: int 
            The number of merges that have already taken place 
            in the hierarchical clustering B. 

        """

        # Relabelling procedure for B

        if j_1 >= self.n: 
            j_1 = self.update_B.pop(j_1)

        if j_2 >= self.n: 
            j_2 = self.update_B.pop(j_2)

        if (self.ctot[j_1] >= self.ctot[j_2] and 
            j_1 >= self.n and j_2 >= self.n): 
            j_1, j_2 = j_2, j_1 

        self.update_B[k+self.n] = j_2

        # Update the column totals and Q 

        ctot1, ctot2 = self.ctot.pop(j_1), self.ctot[j_2]
        self.ctot[j_2] = csum = ctot1 + ctot2
        self.Q += sum([csum**2, -ctot1**2, -ctot2**2]) // 2

        # Procedure used to merge the clusters in B

        c1, c2, st = self.columns.pop(j_1), self.columns[j_2], 0  
        
        for elem in c1: 
 
            if elem not in c2:
                c2[elem] = self.rows[elem][j_2] = self.rows[elem].pop(j_1)

            else: 
                value_1 = self.rows[elem].pop(j_1) 
                value_2 = self.rows[elem][j_2] 
                value_new = value_1 + value_2 
                c2[elem] = self.rows[elem][j_2] = value_new 
                st += sum((value_new**2,-value_1**2, -value_2**2))

        self.columns[j_2] = c2
        self.T += st // 2


    def merge(self, i_1, i_2, j_1, j_2, isize, jsize, k):
        
        """
        Parameters
        ----------

        i_1, i_2 : int
            The labels of the two clusters to be merged in the
            clustering A.

        j_1, j_2 : integers 
            The labels of the two clusters to be merged in the 
            clustering B. 
        
        isize : integer
            The size of the resultant cluster for A. 
        
        jsize : integer 
            The size of the resultant cluster for B. 
        
        k : integer
            Number of merges that have taken place before this merge 

        """

        self.merge_rows(i_1, i_2, isize, k) 
        self.merge_cols(j_1, j_2, jsize, k) 
        return (self.T, self.P, self.Q)


class matching_matrix_proportion():

    

    def __init__(self, sol, n, * args, **kwargs):        
        
        self.rows = {i : {int(j):1} for i, j in enumerate(sol)}
        self.n = n
        self.k = len(Counter(sol)) 
        self.update_A = {}
        self.maximums = np.zeros(self.k)
        self.where = np.zeros(self.k)

    def merge_rows(self, i_1, i_2, isize, k): 
    
        if i_1 >= self.n: 
            i_1 = self.update_A.pop(i_1) 

        if i_2 >= self.n: 
            i_2 = self.update_A.pop(i_2) 

        if (len(self.rows[i_1]) >= len(self.rows[i_2]) and
            i_1 >= self.n and i_2 >= self.n): 
            i_1 , i_2 = i_2, i_1
        
        self.update_A[k + self.n] = i_2 

        r1, r2 = self.rows.pop(i_1), self.rows[i_2]

        # If the maximum is moving then update it's location
        if i_1 in self.where:
            self.where[np.where(self.where == i_1)] = i_2
        
        for elem in r1: 
          
            if elem not in r2:
                r2[elem] = r1[elem]
                    
            else: 
                value_1 = r1[elem]
                value_2 = r2[elem]
                value_new = value_1 + value_2
                r2[elem] = value_new
                if value_new > self.maximums[elem]: 
                    self.maximums[elem] = value_new
                    self.where[elem] = i_2


        self.rows[i_2] = r2
 
    def to_dense(self):

        m = len(self.rows)
        n = self.k 
        M = np.zeros((m, n))
        for i, row in enumerate(self.rows):
            for j, col in self.rows[row].items():
                M[i, j] = self.rows[row][j] 
        return(M)

class matching_matrix_pairs():

    """        
    Class used to store a matching matrix. The matching matrix is used 
    to store information about two clusterings of a set of :math:n 
    objects. We call the first clustering :math:`A` and the second :
    math:`B`. Each row represents a cluster :math:`i` from the 
    clustering :math:`A`. Each column represents a cluster :math:`j` 
    from the clustering :math:`B`. Each element of the matching matrix 
    will then be the number of objects in the cluster :math:`i` that are 
    also in the cluster :math:`j`.

    This version of the matching matrix is designed to be used to
    compare a single clustering to a hierarchy. Hence will take input
    of a particular clustering as well as n the number of objects in
    the clustering. 

    Parameters
    ----------
    n : integer
        The number of objects in the clustering
        
    sol : ndarray
        The clustering to be compared to the hierarchy. 

    """

    def __init__(self, sol, n, * args, **kwargs):        
        
        self.rows = {i : {j:1} for i, j in enumerate(sol)}
        self.rtot = {i : 1 for i, j in enumerate(sol)}
        self.n = n
        c = Counter(sol)
        self.T = 0
        self.P = 0  
        self.Q = (sum([i ** 2 for i in c.values()]) - n) // 2 
        self.update_A = {}

    def merge_rows(self, i_1, i_2, isize, k): 
    
        if i_1 >= self.n: 
            i_1 = self.update_A.pop(i_1) 

        if i_2 >= self.n: 
            i_2 = self.update_A.pop(i_2) 

        if (self.rtot[i_1] >= self.rtot[i_2] and
            i_1 >= self.n and i_2 >= self.n): 
            i_1 , i_2 = i_2, i_1
        
        self.update_A[k + self.n] = i_2 

        rtot1, rtot2 = self.rtot.pop(i_1), self.rtot[i_2]
        self.rtot[i_2] = rsum = rtot1 + rtot2
        self.P += sum([rsum**2, -rtot1**2, -rtot2**2]) // 2
        
        r1, r2, st = self.rows.pop(i_1), self.rows[i_2], 0
        
        for elem in r1: 
          
            if elem not in r2:
                r2[elem] = r1[elem]
 
            else: 
                value_1 = r1[elem]
                value_2 = r2[elem]
                value_new = value_1 + value_2
                r2[elem] = value_new
                st += sum((value_new ** 2, -value_1 ** 2, -value_2 ** 2))

        self.T += st // 2
        self.rows[i_2] = r2

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
        T[k], P[k], Q[k] = m.merge(rows_A[0], rows_A[1], 
                                   rows_B[0], rows_B[1],rows_A[3], rows_B[3],k)
        
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
    
def similarity(T, P, Q, n = None):

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

