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
from numpy.random import default_rng

def flattenList(binnedData):
        flat_list = []
        for sublist in binnedData:
            for item in sublist:
                flat_list.append(item)
        return np.array(flat_list)


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
def learnWeightMatrix(pattern):
    N = 784
    imageVector = pattern.tolist()
    weightMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            weightMatrix[i, j] = (imageVector[i]*imageVector[j])
            if i==j:
                weightMatrix[i, j] = 0
    weightMatrix *= (1/N)
    return weightMatrix

def calculateConfigurationEnergy(weightMatrix, nodeMatrix):

    E = -0.5 * ((np.transpose(nodeMatrix.reshape(784,1)) @ weightMatrix) @ nodeMatrix.reshape(784,1))

    return E

def learnWeightMatrix2(patterns):
    
    

    p1 = (patterns[0])
    p2 = (patterns[1])
    p3 = (patterns[2])

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

# combines two images, randomly selects x proportion of pixels of the 
def combineImages(image1, image2, x):
    newImage = []

    image1 = np.array(image1).ravel()
    image2 = np.array(image2).ravel()

    pixelsToGrab = x * len(image1)
    pixelsToGrab = int(round(pixelsToGrab))

    rng = default_rng()
    indices = rng.choice(len(image1), size=pixelsToGrab, replace=False)

    # loop from 0 to imagelength
    for i in range (len(image1)):
        # if i is in indices, append the index from image1
        exists = i in indices
        if(exists):
            newImage.append(image1[i])
        else:
            newImage.append(image2[i])

    return newImage

N = 784
train_images_raw = np.loadtxt('train_images.csv', delimiter=',')
test_images_raw = np.loadtxt('test_images.csv', delimiter=',')
# split into 784 sized chunks
train_images = np.split(train_images_raw, 3, axis = 0)
test_images = np.split(test_images_raw, 2, axis = 0)

weightMatrix = learnWeightMatrix2(train_images)


# reshape into 2d array

#train_image_01 = combineImages(train_images[0], train_images[1], 0.3)
#train_image_02 = combineImages(train_images[0], train_images[1], 0.3)
#train_image_12 = combineImages(train_images[0], train_images[1], 0.3)

train_image_01_energies = []
mixingFractions = []

for x in range(101):
    print("testing 01 ", x)
    x /= 100
    mixingFractions.append(x)
    train_image_01 = combineImages(train_images[0], train_images[1],x)
    train_image_01 = np.array(train_image_01).ravel()
    train_image_01_energies.append(calculateConfigurationEnergy(weightMatrix, train_image_01.reshape(28,28)).ravel())

train_image_02_energies = []


for x in range(101):
    print("testing 01 ", x)
    x /= 100
   # mixingFractions.append(x)
    train_image_02 = combineImages(train_images[0], train_images[2],x)
    train_image_02 = np.array(train_image_02).ravel()
    train_image_02_energies.append(calculateConfigurationEnergy(weightMatrix, train_image_02.reshape(28,28)).ravel())

train_image_12_energies = []

for x in range(101):
    print("testing 12 ", x)
    x /= 100
   # mixingFractions.append(x)
    train_image_12 = combineImages(train_images[1], train_images[2],x)
    train_image_12 = np.array(train_image_12).ravel()
    trainImageVector = np.reshape(train_image_12, (28,28))
   # title = "combination of ", x*100 , "% training image 0 and ", (1-x)*100 , "% training image 1"
   # plt.title(title)
   # plt.imshow(trainImageVector)
   # plt.show()
    train_image_12_energies.append(calculateConfigurationEnergy(weightMatrix, train_image_12.reshape(28,28)).ravel())




train_image_01_energies = flattenList(train_image_01_energies)
train_image_02_energies = flattenList(train_image_02_energies)
train_image_12_energies = flattenList(train_image_12_energies)


print(train_image_01_energies)
print(train_image_02_energies)





plt.plot(mixingFractions, np.transpose(np.array(train_image_01_energies)), label="images 1 and 2", c = "red")
plt.plot(mixingFractions, np.transpose(np.array(train_image_02_energies)), label="images 1 and 3", c = "green")
plt.plot(mixingFractions, np.transpose(np.array(train_image_12_energies)), label="images 2 and 3", c = "blue")
plt.legend(loc='upper right')
plt.title("Change in Hopfield Network Energy as a Function of Image Merging")
plt.xlabel('ratio of image B to A', fontsize=18)
plt.ylabel('Network energy level', fontsize=16)
plt.show()




doRest = True
if(doRest == True):
    nodes = np.reshape(train_image_01,(28,28))
    plt.imshow(nodes)
    plt.show()

    # create new train_images

    # create a weight matrix
    weightMatrix = learnWeightMatrix2(train_images)

    # now let's try evolving the image
    
    testImageVector = np.reshape(test_images[1], (1,N))
    testImageVector = np.array(testImageVector).ravel()
    h = np.zeros((N))
    #print(np.shape(testImageVector))

    stopNumber = 0
    iteration = 0
    energies = []
    maxPlot = 10000

if(doRest == True):

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
        
        if(iteration % 1000 == 0):
            print("weightMatrix shape: " , np.shape(weightMatrix))
            print("testImageVector shape: " , np.shape(testImageVector.reshape(28,28)))
            
            
            plt.imshow(testImageVector.reshape(28,28))
            titleString = 'Iteration ', iteration, ' of hopfield network on testing image 1 '
            plt.title(titleString)
            #plt.legend(loc='upper right')
            plt.show()
        
            

    

    titleString = 'Final iteration ', stopNumber, ' of hopfield network on testing image 1 '
    plt.title(titleString)
    plt.imshow(testImageVector.reshape(28,28))
    plt.show()
    energies = flattenList(energies)
    # now plot the energy graph
    titleString = 'Minimum hopfield energy over iteration lifetime '
    plt.title(titleString)
    print(energies)
    plt.plot(range(len(energies)), np.transpose(np.array(energies)))
    plt.show()



