import random
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import elephant
import math
import scipy
from scipy.optimize import fsolve
from tqdm import tqdm

def splitDataIntoChunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def getColumn(nodeMatrix, x):
    column = []
    for y in range (28):
        column.append(nodeMatrix[x][y])
    return column

# takes in node data and learns it. outputs a weight matrix
# takes in node data and learns it. outputs a weight matrix
def learnWeightMatrix(pattern, multiplier):
    

    N = 784
    
    im_np = pattern.reshape(1, N)
    # epsidolon is an array of our input pattern
    imageVector = im_np
    #N = len(weightMatrix)
    
    
    weightMatrix = np.zeros((N, N))
    h = np.zeros((N))
    for i in tqdm(range(N)):
        for j in range(N):
            weightMatrix[i, j] = (imageVector[0, i]*imageVector[0, j])
            if i==j:
                weightMatrix[i, j] = 0
  #  weightMatrix *= (1/N)
    print(weightMatrix)

    return weightMatrix

def calculateConfigurationEnergy(weightMatrix, nodeMatrix):

    E = -0.5 * ((np.transpose(nodeMatrix.reshape(784,1)) @ weightMatrix) @ nodeMatrix.reshape(784,1))

    return E

def learnWeightMatrix2(patterns):
    
    

    p1 = (patterns[0])
    print(p1)
    p2 = (patterns[1])
    p3 = (patterns[2])
    epsilon = patterns

    N = 784
    
   # im_np = pattern.reshape(1, N)
    # epsidolon is an array of our input pattern
    #imageVector = im_np
    #N = len(weightMatrix)
    
    
    w = np.zeros((N, N))
    h = np.zeros((N))
    for i in tqdm(range(N)):
        for j in range(N):
            for p in range(3):
                w[i, j] += ((p1[0,i]*p1[0,j]) + (p2[0,i]*p2[0,j]) + (p3[0,i]*p3[0,j]))
            if i==j:
                w[i, j] = 0
    w /= N
    w

    return w




train_images_raw = np.loadtxt('train_images.csv', delimiter=',')
test_images_raw = np.loadtxt('test_images.csv', delimiter=',')
# split into 784 sized chunks
train_images = np.split(train_images_raw, 3, axis = 0)
test_images = np.split(test_images_raw, 2, axis = 0)
# reshape into 2d array




nodes = np.reshape(test_images[0],(28,28))
#plt.imshow(nodes)

#ax2.plot(x, -y)
#plt.show()
nodes = np.reshape(test_images[1],(28,28))

#plt.imshow(nodes)
#plt.show()
#nodes = np.reshape(train_images[2],(28,28))
#ax3.imshow(nodes)
#plt.imshow(nodes)
#plt.show()
# create a weight matrix
plt.show()

# fill the weight matrix with values calculated from the nodematrix
width = 28
height = 28
# find out how many synapic connections we have
connectionCount = width * height - height
# a is the id of the current node we are updating
a = 0
evolutions = 1

# we have loaded an initial train image into nodes
# we learn it with weights
# it should take in a list of patterns

#weightMatrix = learnWeightMatrix(train_images[0], (1/3))
#print(weightMatrix)
#weightMatrix = weightMatrix + learnWeightMatrix(train_images[1], (1/3))
#print(weightMatrix)
#weightMatrix = weightMatrix + learnWeightMatrix(train_images[2], (1/3))
#print(weightMatrix)

weightMatrix = learnWeightMatrix2(train_images)
#print(weightMatrix)
#print(len(weightMatrix), " LENGTH weightMatrix")
#print( "weeeeights ", len(np.array(weightMatrix).ravel()))
# now let's try evolving the image
N = 784
testImageVector = np.reshape(test_images[1], (1,N))
testImageVector = np.array(testImageVector).ravel()
h = np.zeros((N))
#print(np.shape(testImageVector))

stopNumber = 0
iteration = 0
energies = []
maxPlot = 10000
for iteration in range(maxPlot):
    stopNumber = iteration
    lastPattern = testImageVector
    i = np.random.randint(N)
    testImageVector[i] = np.sign(weightMatrix[i].T @ testImageVector)
    energies.append(calculateConfigurationEnergy(weightMatrix, testImageVector.reshape(28,28)).ravel())
    hamming0 = scipy.spatial.distance.hamming(testImageVector,train_images[0])
    hamming1 = scipy.spatial.distance.hamming(testImageVector,train_images[1])
    hamming2 = scipy.spatial.distance.hamming(testImageVector,train_images[2])
    if(hamming0 == 0 or hamming1 == 0 or hamming2 == 0):
        stopNumber = iteration
        break
   # print(hamming)
    
    if(iteration == 0 or iteration == 1000 or iteration == 3000):
        print("weightMatrix shape: " , np.shape(weightMatrix))
        print("testImageVector shape: " , np.shape(testImageVector.reshape(28,28)))
        
        
        plt.imshow(testImageVector.reshape(28,28))
        titleString = 'Iteration ', iteration, ' of hopfield network on testing image 1 '
        plt.title(titleString)
        #plt.legend(loc='upper right')
        plt.show()
    
        

def flattenList(binnedData):
    flat_list = []
    for sublist in binnedData:
        for item in sublist:
            flat_list.append(item)
    return np.array(flat_list)

titleString = 'Final iteration ', stopNumber, ' of hopfield network on testing image 1 '
plt.title(titleString)
plt.imshow(testImageVector.reshape(28,28))
plt.show()
energies = flattenList(energies)
# now plot the energy graph
titleString = 'Image 2 Hopfield Network Energy over Iteration lifetime '
plt.title(titleString)
plt.xlabel('Evolution Iteration', fontsize=18)
plt.ylabel('Energy Value', fontsize=18)
print(energies)
plt.plot(range(len(energies)), np.transpose(np.array(energies)))
plt.show()



