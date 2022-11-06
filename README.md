# QR Decomposition Algorithm

# What is it?

Predictive analysis using Least Square Approximation by utilizing QR Decomposition Algorithm.

This is performed on 3 case studies:
- Effects of alcohol on happiness, 
- Housing prices in the US,
- Factors of happiness

Read our report to learn more about this project [here](https://github.com/albertarielw/qr-decomposition-prediction/blob/main/Group%20Report.pdf).

# Sample Implementation

Here is a sample implementation of QR Decomposition Algorithm

```python
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

    return (Q, R)     
```

Here is the link for more implementation [link](https://github.com/albertarielw/qr-decomposition-prediction/blob/main/code/QR%20Decomposition%20(Our%20Own%20Implementation).py)
