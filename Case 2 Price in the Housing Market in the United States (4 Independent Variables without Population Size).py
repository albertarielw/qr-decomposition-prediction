### Multiple Independent Variables ####
# The dependent variables used are: Average Income in the Area, Average House Age in the Area, Average Number of Rooms in the Area, Average Number of Bedrooms in the Area, Population in the Area

import numpy as np
import matplotlib.pyplot as plt
import math
import csv

file = open("Case 2 Price in the Housing Market in the United States.csv")

csvreader = csv.reader(file)

header = []
header = next(csvreader)

Y = []
X = []
X_axis = []

for row in csvreader:
    # Dependent Variable: Prices
    Y.append([float(row[5])]) 
    # Independent Variables:
    # 1. Average Income in the Area
    # 2. Average House Age in the Area
    # 3. Average Number of Rooms in the Area
    # 4. Average Number of Bedrooms in the Area
    X.append([float(row[0]),
              float(row[1]),
              float(row[2]),
              float(row[3]),
              #float(row[4])
              ])
    # Average Income in the Area is chosen as X Axis
    X_axis.append([float(row[0])])

# Convert X and Y arrays into matrices
# so that numpy matrix operations can be performed 
X = np.array(X) 
Y = np.array(Y)

# Perform QR Decomposition Algorithm
Q, R = np.linalg.qr(X)

R_inverse = np.linalg.inv(R) # Find Inverse of R
Q_transpose = Q.T # Find Transpose of Q
B = np.matmul(np.matmul(R_inverse, Q_transpose), Y) # B = Inverse of R * Transpose of Q * Y 
Y_Predicted = np.matmul(X,B) # Y = X * B

plt.title("Price against Average Income\n in the United States Housing Market")
plt.xlabel("Average Income (USD)")
plt.ylabel("Price (1 Million USD)")
plt.scatter(X_axis,Y,color="red", label="Actual Price")
plt.scatter(X_axis,Y_Predicted,color="black", label="Predicted Price")
plt.legend(loc = "upper left")
plt.show()

MPE = 0.0 # Mean Percentage Error
MAE = 0.0 # Mean Absolute Error
MSE = 0.0 # Mean Squared Error
RMSE = 0.0 # Root Mean Squared Error

for i in range(0, 5000):
    MPE += abs((Y[i][0] - Y_Predicted[i][0]) / Y[i][0]) * 100
    MAE += abs(Y[i][0] - Y_Predicted[i][0])
    MSE += (Y[i][0] - Y_Predicted[i][0])**2

MPE = MPE / 5000.0
MAE = MAE / 5000.0
MSE = MSE / 5000.0 
RMSE = MSE ** (0.5)

R_Squared = 0.0 # Coefficient of Determination
Adjusted_R_Squared = 0.0
Y_Mean = 0.0

for i in range(0, 5000):
    Y_Mean += Y[i][0]

Y_Mean = Y_Mean / 5000

SSR = 0.0 # Sum Squared Regression
TSS = 0.0 # Total Sum of Squares

for i in range(0, 5000):
    SSR += (Y[i][0] - Y_Predicted[i][0])**2
    TSS += (Y[i][0] - Y_Mean)**2

R_Squared = 1 - (SSR / TSS)
Adjusted_R_Squared = 1 - (1 - R_Squared**2) * (5000 - 1) / (5000 - 4 - 1)

print(MPE)
print(MAE)
print(MSE)
print(RMSE)
print(R_Squared)
print(Adjusted_R_Squared)
