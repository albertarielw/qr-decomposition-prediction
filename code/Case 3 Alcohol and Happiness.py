### Multiple Independent Variables ####
# The dependent variables used are: Average Income in the Area, Average House Age in the Area, Average Number of Rooms in the Area, Average Number of Bedrooms in the Area, Population in the Area

import numpy as np
import matplotlib.pyplot as plt
import math
import csv

file = open("Case 3 Alcohol and Happiness.csv")

csvreader = csv.reader(file)

header = []
header = next(csvreader)

Y = []
X = []
X_axis = []

for row in csvreader:
    # Dependent Variable: Happiness Score
    Y.append([float(row[3])]) 
    # Independent Variables:
    # 1. Beer Per Capita
    # 2. Spirit Per Capita
    # 3. Wine Per Capita
    X.append([float(row[6]), float(row[7]), float(row[8])])
    # Average Income in the Area is chosen as X Axis
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

plt.title("Happiness Score against Beer Per Capita")
plt.xlabel("Beer Per Capita")
plt.ylabel("Happiness Score")
plt.scatter(X_axis,Y,color="red", label="Actual Score")
plt.scatter(X_axis,Y_Predicted,color="black", label="Predicted Score")
plt.legend(loc = "upper left")
plt.show()

MPE = 0.0 # Mean Percentage Error
MAE = 0.0 # Mean Absolute Error
MSE = 0.0 # Mean Squared Error
RMSE = 0.0 # Root Mean Squared Error
for i in range(0, 122):
    MPE += abs((Y[i][0] - Y_Predicted[i][0]) / Y[i][0]) * 100
    MAE += abs(Y[i][0] - Y_Predicted[i][0])
    MSE += (Y[i][0] - Y_Predicted[i][0])**2

MPE = MPE / 122.0
MAE = MAE / 122.0
MSE = MSE / 122.0 
RMSE = MSE ** (0.5)

print(MPE)
print(MAE)
print(MSE)
print(RMSE)

