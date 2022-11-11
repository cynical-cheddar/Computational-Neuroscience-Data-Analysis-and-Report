#%%

# In this version of the code I discovered how overfitted models use very high coefficients of fit
import random
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from utilities import load_points_from_file, view_data_segments

# Global Settings
repetitions = 100
bestOrder = 3 
# Initialise the current order and best fitting order variables
maxOrder = 3


def square_error(y, y_hat):
    x = np.sum((y - y_hat) ** 2)
    return x

# Find the set of least squares coefficients
def least_squares(xs, ys, order):
    ones = np.ones(xs.shape)
    
    if(order == 0):
      X = np.column_stack((ones, np.sin(xs))) 
    else:
      X = np.column_stack((ones, xs))
      for (i) in range (order -1):
        X = np.column_stack((X, xs**(i+2)))    
    v = np.linalg.inv(X.T @ X) @ X.T @ ys
    return v

# Splits xs and ys into sets of 20
def getDataSets(xs,ys):
  xsData = list(chunks(xs, 20))
  ysData = list(chunks(ys, 20))
  return xsData, ysData

# Yield function splits up an array into lengths of size n
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Find the output set of a regression function as well as its coefficients
def calculateRegressionY(xs, ys, order):
  coefficients = least_squares(xs, ys, order)
  i = 0
  regressionY = np.zeros(np.shape(ys))
  # Order zero corresponds to sin
  if(order == 0):
    regressionY = np.sin(xs) * coefficients[1] + coefficients[0]
  else:
    for j in range (np.size(coefficients)):
      if(j == 0):
        regressionY = xs * coefficients[1] + coefficients[0]
      if(j > 1):
        regressionY = regressionY + np.array(coefficients[j])*(xs**(j))
  return regressionY, coefficients
  
# Generates the output of a function given the set of xs and the coefficients.
def generateYvalues(xs, ys, coefficients):
  i = 0
  generatedYs = np.zeros(np.shape(ys))
  for j in range (np.size(coefficients)):
    if(j == 0):
      generatedYs = xs * coefficients[1] + coefficients[0]
    if(j > 1):
      generatedYs = generatedYs + np.array(coefficients[j])*(xs**(j))
  return generatedYs




def kFoldValidationError(xDataSet, yDataset, j):
    # Defines our various datasets for cross validation
    # Cross Validation Settings
    sumError = 0
    validationSize = 5
    trainingSize = 20 - validationSize

    # iterate k times over k randomly chosen sets. Find the sum of errors
    for i in range (repetitions):
      # Shuffle our datasets simultaneously with the same seed.
      # This preserves relative ordering
      orderedShufffle = np.arange(xDataSet.shape[0])
      np.random.shuffle(orderedShufffle)
      xDataSet = xDataSet[orderedShufffle]
      yDataset = yDataset[orderedShufffle]
      # Define our newly shuffled sets
      xTrainingSet = xDataSet[:trainingSize]
      yTrainingSet = yDataset[:trainingSize]
      xValidationSet = xDataSet[trainingSize:]
      yValidationSet = yDataset[trainingSize:]
      # Calculates the least squares regression using our current training set
      ysRegression, coefficients = calculateRegressionY (xTrainingSet, yTrainingSet, j)
      # generate a set of ys corresponding to each x in xValidationSet
      ysOnLine = generateYvalues(xValidationSet, yValidationSet, coefficients)
      sumError += square_error(yValidationSet, ysOnLine)
    return sumError



## THE MAIN FUNCTION ~~
##############################################################
xs, ys = load_points_from_file("train_data/" + sys.argv[1])
plot = False
if(len(sys.argv) > 2):
  if(sys.argv[2] == "--plot"):
    plot = True
xsSets, ysSets = getDataSets(xs, ys)

# Our data has been split into chunks of 20 points
# We may now iterate through each subset to find the most
# appropriate model.
i = 0
finalError = 0
for xDataSet in xsSets:
  # Set the current smallest error to infinity - reduce it once we find a smaller error
  error = math.inf
  # Iterate over each polynomial order
  for j in range (0, maxOrder + 1):
    # Sine testing
    if(j == 0):
      sinY, a = calculateRegressionY(xDataSet, ysSets[i], 0)
      newError = square_error(sinY, ysSets[i]) * repetitions
    # Polynomial testing
    elif(j != 2):
      newError = kFoldValidationError(xDataSet, ysSets[i], j)    
    if (newError < error):
      error = newError
      bestOrder = j 
  #  print("Error: ", newError, "  order: ", j)
    
  # We have found the best order for this section of points. Plot it.
  finalY, a = calculateRegressionY(xDataSet, ysSets[i], int(bestOrder))
  finalError += square_error(finalY, ysSets[i])
  plt.plot(xDataSet, finalY) 

  
  i += 1

# Sum the error of each section and output it
print(finalError)
if(plot):
  view_data_segments(xs,ys)
    


# %%

# %%
