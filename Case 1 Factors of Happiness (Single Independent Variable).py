### Multiple Independent Variables ####
# The dependent variables used are: Average Income in the Area, Average House Age in the Area, Average Number of Rooms in the Area, Average Number of Bedrooms in the Area, Population in the Area

import numpy as np
import matplotlib.pyplot as plt
import math
import csv

file = open("Case 1 Factors of Happiness.csv")

csvreader = csv.reader(file)

header = []
header = next(csvreader)

Y = []
X = []
X_axis = []

for row in csvreader:
    # Dependent Variable: Happiness Score
    Y.append([float(row[3])]) 
    # Independent Variables: Economy (GDP Per Capita relative to the United States)
    X.append([float(row[6])])
    # Economy (GDP Per Capita relative to the United States)
    X_axis.append([float(row[6])])

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

plt.title("Happiness Score against \nGDP Per Capita relative to the United States")
plt.xlabel("GDP Per Capita relative to the United States")
plt.ylabel("Happiness Score")
plt.scatter(X_axis,Y,color="red", label="Actual Happiness Score")
plt.scatter(X_axis,Y_Predicted,color="black", label="Predicted Happiness Score")
plt.legend(loc = "upper left")
plt.show()

MPE = 0.0 # Mean Percentage Error
MAE = 0.0 # Mean Absolute Error
MSE = 0.0 # Mean Squared Error
RMSE = 0.0 # Root Mean Squared Error

for i in range(0, 157):
    MPE += abs((Y[i][0] - Y_Predicted[i][0]) / Y[i][0]) * 100
    MAE += abs(Y[i][0] - Y_Predicted[i][0])
    MSE += (Y[i][0] - Y_Predicted[i][0])**2

MPE = MPE / 157
MAE = MAE / 157
MSE = MSE / 157
RMSE = MSE ** (0.5)

R_Squared = 0.0 # Coefficient of Determination
Adjusted_R_Squared = 0.0
Y_Mean = 0.0

for i in range(0, 157):
    Y_Mean += Y[i][0]

Y_Mean = Y_Mean / 157

SSR = 0.0 # Sum Squared Regression
TSS = 0.0 # Total Sum of Squares

for i in range(0, 157):
    SSR += (Y[i][0] - Y_Predicted[i][0])**2
    TSS += (Y[i][0] - Y_Mean)**2

R_Squared = 1 - (SSR / TSS)
Adjusted_R_Squared = 1 - (1 - R_Squared**2) * (157 - 1) / (157 - 1 - 1)

print(MPE)
print(MAE)
print(MSE)
print(RMSE)
print(R_Squared)
print(Adjusted_R_Squared)
