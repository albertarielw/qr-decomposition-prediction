import numpy as np
import math

"""QR Decomposition of matrix A using Gram Schmidt"""
def QR_Decomposition_Gram_Schmidt(A):                           
    (r, c) = np.shape(A)                                        # r and c are number of rows and columns of A respectively

    Q = np.empty([r, r])                                        # Initialize an empty orthogonal matrix Q
    n = 0                                                       # Initialize column counter n to compute projection in Gram Schmidt process

    # Compute orthogonal matrix Q.
    for a in A.T:                                               # For every row a in Transpose of a
        u = np.copy(a)                                          # Create a matrix u which is a copy of a

        # Compute Gram Schmidt Process
        for i in range(0, n):                             
            projection = np.dot(np.dot(Q[:, i].T, a), Q[:, i])  # Compute the i-th projection in the Gram Schmidt Process
            u = u - projection                                  # Substract u by the i-th projection in the Gram Schmidt Process

        e = u / np.linalg.norm(u)                               # Compute othonormal basis e
        Q[:, n] = e                                             # For every row, from column n to the end assign the value of e

        n += 1                                                  # Increase columns counter.

    # Compute upper triangular matrix R.
    R = np.matmul(Q.T, A) 

    return (Q, R)                                               # Return a tuple which consists of the Q and R matrices

"""QR Decomposition of matrix A using Householder Reflection"""
def QR_Decomposition_Householder_Reflection(A):
    (r, c) = np.shape(A)                                        # r and c are number of rows and columns of A respectively

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(r) # Define Q as r x r identity matrix
    R = np.copy(A)  # Define R as a copy of A

    # Iterative over column sub-vector and compute Householder matrix to zero-out lower triangular matrix entries.
    for i in range(r - 1):
        x = R[i:, i]

        e = np.zeros_like(x)
        e[0] = math.copysign(np.linalg.norm(x), -A[i, i])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_i = np.identity(r)
        Q_i[i:, i:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_i, R)
        Q = np.dot(Q, Q_i.T)

    return (Q, R)                                               # Return a tuple which consists of the Q and R matrices

"""QR Decomposition of matrix A using Givens Rotation"""
def QR_Decomposition_Givens_Rotation(A):
    # Define a helper function to compute Givens Rotation of matrix entries x and y
    def Givens_Rotation(x, y):
        length = math.sqrt(x**2 + y**2)                         # Length of vector (x, y)
        c = x/length                                            # Compute cos (theta), c
        s = -y/length                                           # Compute sin (theta), s

        return (c, s)                                           # Return a tuple of c and s
    
    (r, c) = np.shape(A)                                        # r and c are number of rows and columns of A respectively

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(r)                                          # Define Q as r x r identity matrix
    R = np.copy(A)                                              # Define R as a copy of A

    (rows, cols) = np.tril_indices(r, -1, c)
    # np.tril_indices(r, -1, c) returns the indices for the lower-triangle of an (n, m) array with a diagonal offset of -1
    # Then, initialize rows and cols with the returned array of indices along one dimension of the array
    
    for (row, col) in zip(rows, cols):                          # Iterate over the triangular matrix created using zip(r2, c2)

        # Compute Givens rotation matrix and zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = Givens_Rotation(R[col, col], R[row, col])

            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)                                               # Return a tuple which consists of the Q and R matrices



import csv

file = open("Dataset.csv")

csvreader = csv.reader(file)

header = []
header = next(csvreader)

Y = []
X = []
for row in csvreader:
    Y.append([float(row[3])])
    X.append([float(row[5])])

X = np.array(X)
Y = np.array(Y)

(Q, R) = QR_Decomposition_Givens_Rotation(X)

#R_inverse = np.linalg.inv(R)
#Q_transpose = Q.T
"B = np.matmul(np.matmul(R_inverse, Q_transpose), Y)"













































