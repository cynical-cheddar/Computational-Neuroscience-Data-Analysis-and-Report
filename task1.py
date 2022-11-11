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

def getMeanInterval(spikeTimesInBin):
    
    intervals = []

    for i in range (len(spikeTimesInBin)):
        j = (len(spikeTimesInBin)) - i -1
        if(j > 0):
            interval = spikeTimesInBin[j] - spikeTimesInBin[j-1]
            intervals.append(interval)  
    return np.mean(intervals)

def getIntervals(spikeTimesInBin):
    
    intervals = []
    for i in range (len(spikeTimesInBin)):
        j = (len(spikeTimesInBin)) - i -1
        if(j > 0):
            interval = spikeTimesInBin[j] - spikeTimesInBin[j-1]
            intervals.append(interval)  
    return intervals

def coefficentOfVariationBinned(neuronDataBins, timeBin):
    # get dataset of all intervals
   # neuronDataBins = getTimeBins(neuronDataBins, timeBin)
    neuronDataBins = sortIntoBins(neuronDataBins, timeBin)
    binIntervals = []
    # get set of intervals 
    for dataBin in neuronDataBins:
        if(len(dataBin) >= 2):
            binIntervals.append(getIntervals(dataBin))
    # cv = sd/mean
    binIntervals = flattenList(binIntervals)
    mean = np.mean(binIntervals)
    sd = np.std(binIntervals)
    
    return (sd/mean)

def coefficentOfVariation(neuronData):
    # get dataset of all intervals
    intervals = []
    data = np.array(neuronData)

    for i in range (len(neuronData)):
        j = (len(neuronData)) - i -1
        if(j > 0):
            interval = data[j] - data[j-1]
            intervals.append(interval)
    # cv = sd/mean
    mean = np.mean(intervals)
    sd = np.std(intervals)
    print("cv - mean: ", mean, " sd: ", sd)
    return (sd/mean)

def sortBin(data, min, max):
    bin = []
    newMin = min
    if(min !=0):
        min+=1
    for x in range(min, len(data)):
        if(min<=data[x] and data[x] <max):
            bin.append(data[x])
            newMin = x
        if(data[x]>max):
            break

    return bin, newMin





## QUESTIONS


# splits dataset into bins and returns a list of the length of each bin
def countIntoBinsNoZero(neuronData, timeBin):
    data = neuronData.tolist()
    
    binCount = []
    size = len(data)
    numberOfBins = int(np.ceil(data[size-1]/timeBin))
    for i in range (numberOfBins):
        count = countBin(data, i*timeBin, (i+1)*timeBin)
        if(count > 0):
           binCount.append(count)
    return binCount



def getBinIntervals(neuronData, timeBin):
    intervals = []


    size = len(neuronData)
    n = int(np.ceil(neuronData[size-1]/timeBin))

    for i in range(n):
        intervals.append((i+1) * timeBin)
    return intervals


# returns list of bin sizes
def countBins(bins):
    binCounts = []
    for x in bins:
        binCounts.append(len(x))
    return binCounts


# efficiently returns size of each bin from raw dataset
def countBin(data,minVal, maxVal):
    count = 0
    for x in data:
        if(x >= minVal and x<maxVal):
            count += 1
        
    return count

def countIntoBins(neuronData, timeBin):
    data = neuronData.tolist()
    
    binCount = []
    size = len(data)
    numberOfBins = int(np.ceil(data[size-1]/timeBin))
   # print("number of bins: ", numberOfBins)
    for i in range (numberOfBins):
        count = countBin(data, i*timeBin, (i+1)*timeBin)
        binCount.append(count)
    return binCount

def sortIntoBins(data,timeBin):
    data = data.tolist()
    binList = []
    size = len(data)
    n = int(np.ceil(data[size-1]/timeBin))
    minimum = 0
    for i in range(n):
        bin1, minimum = sortBin(data,minimum,(i+1) * timeBin)
        binList.append(bin1)
    return binList

# takes a flattened list of data, the new time bin, the original time bin (1000ms),
# our list of trial ids, as well as the set that we want to filter
def sortIntoBinsWithId(data,timeBin, originalTimeBin, ids, wantedSet):
    valuesPerId = int(originalTimeBin / timeBin)

    data = data.tolist()
    binList = []
    size = len(data)
    n = int(np.ceil(data[size-1]/timeBin))
    minimum = 0
    j = 0
    k = 0
    for i in range(n):
        bin1, minimum = sortBin(data,minimum,(i+1) * timeBin)
        if(ids[k] == wantedSet):
            binList.append(bin1)
        j += 1
        if(j >= valuesPerId):
            k += 1
            j = 0
    # return a list of bins of size timeBin
    return binList

def sortIntoBinsNoZero(data, timeBin):
    data = data.tolist()
    binList = []
    size = len(data)
    n = int(np.ceil(data[size-1]/timeBin))
    minimum = 0
    for i in range(n):
        bin1, minimum = sortBin(data,minimum,(i+1) * timeBin)
        if(len(bin1)>0):
            binList.append(bin1)

         
    return binList

def calculateFanoFactor(neuronData, timeBin):
    # get number of spikes in time bin
    print("neuronData length: ", len(neuronData))
    binCounts = np.array(countIntoBins(neuronData,timeBin))
    print("binCounts length: ", len(binCounts))
    mean = np.mean(binCounts)
    var = np.var(binCounts)
    ff = var/mean

   # print(binCounts)
    #print("mean count in bin: ", mean)
    return ff

# calculate the fano factor with a pre-binned set
def calculateFanoFactorPreBinned(bins):
    # get number of spikes in time bin
    # get array of number of spikes in each bin
    binCounts = np.array(countBins(bins))
    mean = np.mean(binCounts)
    var = np.var(binCounts)
    ff = var/mean
    return ff

# splits a dataset into time bins (inefficiently)
def getTimeBins(neuronData, intervalSize):
    curIntervalLower = 0
    curIntervalUpper = intervalSize
    maxValue = neuronData[-1]
    # get n time bins based on interval size
    timeBins = []
    # get n timebins while the max value < maxValue
    while(curIntervalUpper <= maxValue + intervalSize):
        timeBins.append(getTimeBin(neuronData, curIntervalLower, curIntervalUpper))
        curIntervalLower += intervalSize
        curIntervalUpper += intervalSize
    return timeBins

# get a time bin subset from a dataset
def getTimeBin(neuronData, timeBinLower, timeBinUpper):
    timeBin = []
    for spike in neuronData:
        if(spike >= timeBinLower and spike < timeBinUpper):
            timeBin.append(spike)
        
    return timeBin

def flattenList(binnedData):
    flat_list = []
    for sublist in binnedData:
        for item in sublist:
            flat_list.append(item)
    return np.array(flat_list)

def solveGaussianIntersection(u1,u2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = u2/(std2**2) - u1/(std1**2)
  c = u1**2 /(2*std1**2) - u2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

def decodeSpikeCountMorePerfect(x, dataset1, dataset2):


   # print("Comparing boundary and bin count: " , boundary[0], " ", x)
    if(x > 16):
        return (1)
    else:
        return (0)
    return

# determines the trial ID based on counted bins for a pair of neurons
def decodeSingleTrialSpikePair(xa, xb):
    if(xa - xb - 6 > 0):
        return 1
    else:
        return 0

# decodes from a single neuron
def decodeSpikeCountPerfect(x, dataset1, dataset2):
    u1 = np.mean(dataset1)
    u2 = np.mean(dataset2)

    o1 = np.std(dataset1)
    o2 = np.std(dataset2)

    

    #P(correct) = P(N(u1, o) > N(u2,o))
    # = cumulativeDistrubution * (d' / root 2)
    # we need to get the decision boundary between the two gaussians
    # this can be found when x is between the means
    boundary = solveGaussianIntersection(u1, u2, o1, o2)

    ex1 = (1/(o1 * np.sqrt(2*np.pi))) * (np.e ** -0.5*(((x - u1)/o1)**2))
    ex2 = (1/(o2 * np.sqrt(2*np.pi))) * (np.e ** -0.5*(((x - u2)/o2)**2))
    # now we have our expected values, return true or false

   # print("Comparing boundary and bin count: " , boundary[0], " ", x)
    if(x > boundary[0]):
        return (1)
    else:
        return (0)
    return

def decodeSpikeCount(x, customBoundary):
  #  u1 = np.mean(dataset1)
 #   u2 = np.mean(dataset2)

 #   o1 = np.std(dataset1)
 #   o2 = np.std(dataset2)

    #P(correct) = P(N(u1, o) > N(u2,o))
    # = cumulativeDistrubution * (d' / root 2)
    # we need to get the decision boundary between the two gaussians
    # this can be found when x is between the means
    #boundary = solveGaussianIntersection(u1, u2, o1, o2)

   # ex1 = (1/(o1 * np.sqrt(2*np.pi))) * (np.e ** -0.5*(((x - u1)/o1)**2))
  #  ex2 = (1/(o2 * np.sqrt(2*np.pi))) * (np.e ** -0.5*(((x - u2)/o2)**2))
    # now we have our expected values, return true or false

  #  print("Comparing boundary and bin count: " , customBoundary, " ", x)
    if(x > customBoundary):
        return (1)
    else:
        return (0)
    return
    

def calculateDeePrimeBinned(dataset1, dataset2):
    # get mean of one dataset
    u1 = np.mean(dataset1)
    u2 = np.mean(dataset2)

    var1 = np.var(dataset1)
    var2 = np.var(dataset2)

    #deePrime = abs(u1-u2)/(np.sqrt(0.5 * (var1 + var2)))
    deePrime = (np.sqrt(2) * abs(u1 - u2))/np.sqrt(var2+var1)
    return deePrime

# assumes lists are same length. splits bin set into two
def splitDataByTrialId(neuronData, trialIDs):
    neuronData_0 = []
    neuronData_1 = []
    print("splitting data -neuronData bin length ", len(neuronData))
    for spikeBin, trial_ID in zip(neuronData, trialIDs):
       if(trial_ID == 0):
           neuronData_0.append(spikeBin)
       else:
           neuronData_1.append(spikeBin)
    print("split data -neuronData_0 length ", len(neuronData_0), " ", len(neuronData_1))
    return (neuronData_0), (neuronData_1)




# take bin spike counts
def plotOptimalBoundaryGraph(neuronData_0_counts, neuronData_1_counts):
    optimalBoundary = 0
    truePositiveHitRates = []
    trueNegativeHitRates = []
    perfectNegativeHitRates = []
    morePerfectNegativeHitRates = []
    boundaryAxis = []
    totalCorrectRate = []

    for i in range (40):
        boundaryAxis.append(i)
        truePositives = []
        perfectNegatives = []
        morePerfectNegatives = []
        trueNegatives = []
        correctCounts = []

        for count0 in neuronData_0_counts:
            if(count0>0):
                count0decision = decodeSpikeCount( count0, i)
                if(count0decision == 0):
                    trueNegatives.append(1)
                    correctCounts.append(1)
                    
                else:
                    correctCounts.append(0)
               

        for count1 in neuronData_1_counts:
            if(count1 > 0):
                count1decision = decodeSpikeCount( count1, i)
                if(count1decision == 1):
                    truePositives.append(1)
                    correctCounts.append(1)
                else:
                    correctCounts.append(0)

        totalCorrectRate.append(np.mean(correctCounts))
        correctCounts = []
        truePositiveHitRates.append(np.sum(truePositives)/len(neuronData_1_counts))
        trueNegativeHitRates.append(np.sum(trueNegatives)/len(neuronData_0_counts))
       
       # perfectNegativeHitRates.append(np.mean(perfectNegatives))
        #morePerfectNegativeHitRates.append(np.mean(morePerfectNegatives))

    plt.plot(boundaryAxis, truePositiveHitRates, c = "blue", label= "True Positive Detection Rate")
    plt.plot(boundaryAxis, trueNegativeHitRates, c = "red", label= "True Negative Detection Rate")
  #  plt.plot(boundaryAxis, perfectNegativeHitRates, c = "green")
#plt.plot(boundaryAxis, morePerfectNegativeHitRates, c = "purple", label=["morePerfectNegativeHitRates"])
    
    plt.plot(boundaryAxis, totalCorrectRate, c = "green", label= "Total Correct Rate")

    # loop through the total correct rates to find the highest index
    max_value = max(totalCorrectRate)
    optimalBoundary = totalCorrectRate.index(max_value)

    plt.xlabel('Spike boundary (spikes/trial)')
    plt.ylabel('Classification Rate')
    plt.title('Decoder Classification Rates as a Function of Spike Boundary')
    plt.legend(loc='upper right')

    plt.show()
    return optimalBoundary, max(totalCorrectRate)
   # print(np.mean(morePerfectNegativeHitRates))

q1 = False
q2 = True
q3 = True
q4 = True

# Q1 a
# calculate coefficient of variation
neuronData = np.loadtxt('neuron_A.csv', delimiter=',')
cv = coefficentOfVariation(neuronData)
print("neuron_A fano factor:: " , calculateFanoFactor(neuronData, 1000))
if(q1):
    print("neuron_A coefficient of variation: " , cv)

# Q1 b
# calculate fano factor for time bins

if(q1):
    print("bin 1 fano factor: " , calculateFanoFactor(neuronData, 100))
    print("bin 2 fano factor: " , calculateFanoFactor(neuronData, 200))
    print("bin 3 fano factor: " , calculateFanoFactor(neuronData, 500))
    print("bin 4 fano factor: " , calculateFanoFactor(neuronData, 1000))




   


def splitDataIntoChunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Q1 c

# record 0 - dot with no motion
# record 1 - moving random dot stimulus
# separate trial data into two sets based on trial id


# split data into 1000ms bins
binsize = 1000
neuronDataBins = sortIntoBins(neuronData, binsize)

# split bins by ID 0 and 1
trialData = np.loadtxt('trial_ID.csv', delimiter=',')
neuronDataBins_0, neuronDataBins_1 = splitDataByTrialId(neuronDataBins, trialData)

# now de-bin the neuron data
neuronData_0 = flattenList(neuronDataBins_0)
neuronData_1 = flattenList(neuronDataBins_1)


# do calculations
print("data length after flattening: ", len(neuronData_0) + len(neuronData_1))


if(q1 == True):

    print ("------------------------")

    print("neuronData ff: " , calculateFanoFactorPreBinned(sortIntoBins(neuronData, 100))  ,"(binsize 100)")
    print("neuronData ff: " , calculateFanoFactorPreBinned(sortIntoBins(neuronData, 200)) ," (binsize 200)")
    print("neuronData ff: " , calculateFanoFactorPreBinned(sortIntoBins(neuronData, 500)) , " (binsize 500)")
    print("neuronData ff: " , calculateFanoFactorPreBinned(sortIntoBins(neuronData, 1000)),"cv: ", coefficentOfVariationBinned(neuronData, 1000) , " (binsize 1000)")

    print("neuronData_0 ff: " , calculateFanoFactorPreBinned(sortIntoBinsWithId(neuronData_0, 100, 1000, trialData, 0))  ,"(binsize 100)")
    print("neuronData_0 ff: " , calculateFanoFactorPreBinned(sortIntoBinsWithId(neuronData_0, 200, 1000, trialData, 0)) ," (binsize 200)")
    print("neuronData_0 ff: " , calculateFanoFactorPreBinned(sortIntoBinsWithId(neuronData_0, 500, 1000, trialData, 0)) , " (binsize 500)")
    print("neuronData_0 ff: " , calculateFanoFactorPreBinned(neuronDataBins_0),"cv: ", coefficentOfVariationBinned(neuronData_0, 1000) , " (binsize 1000)")


    print("neuronData_1 ff: " , calculateFanoFactorPreBinned(sortIntoBinsWithId(neuronData_1, 100, 1000, trialData, 1)) ," (binsize 100)")
    print("neuronData_1 ff: " , calculateFanoFactorPreBinned(sortIntoBinsWithId(neuronData_1, 200, 1000, trialData, 1)) ," (binsize 200)")
    print("neuronData_1 ff: " , calculateFanoFactorPreBinned(sortIntoBinsWithId(neuronData_1, 500, 1000, trialData, 1)) ," (binsize 500)")
    print("neuronData_1 ff: " , calculateFanoFactorPreBinned(neuronDataBins_1), "cv: ", coefficentOfVariationBinned(neuronData_1, 1000) ," (binsize 1000)")

if(q2 == True):

  #  size = len(neuronData_0)
  #  numberOfBins = int(np.ceil(neuronData_0[size-1]/1000))
    # count into bins
    neuronData_counts = countIntoBins(neuronData, 1000)
  #  neuronData_0_counts = countIntoBinsNoZero(neuronData_0, 1000)
    neuronData_0_counts = countBins(sortIntoBinsWithId(neuronData_0, 1000, 1000, trialData, 0))
    neuronData_1_counts = countBins(sortIntoBinsWithId(neuronData_1, 1000, 1000, trialData, 1))
    #neuronData_0_counts_with_zero = countIntoBins(neuronData_0, 1000)
   # neuronData_1_counts_with_zero = countIntoBins(neuronData_1, 1000)

    binIntervals_0 = getBinIntervals(neuronData_0, 1000)
    binIntervals_1 = getBinIntervals(neuronData_1, 1000)

    # count number of spikes in each trial

    # get amount of trials (count of bins)
    print("Total amount of neuronData_0 trials" , len(neuronData_0_counts))
    print("Total amount of neuronData_1 trials" , len(neuronData_1_counts))
    # plot both histograms
    plt.title("Spike Count for Non-Moving Stimulus per Trial")
    plt.bar(list(range(0, len(neuronData_0_counts))), neuronData_0_counts)
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Spike Count in Trial', fontsize=18)
    plt.ylim(0,50)
    plt.show()

    plt.title("Spike Count for Moving Stimulus per Trial")
    plt.bar(list(range(0, len(neuronData_1_counts))), neuronData_1_counts)
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Spike Count in Trial', fontsize=18)
    plt.ylim(0,50)
    plt.show()



    # now plot the histogram
    plt.title("Histogram of Trial Spike Count Sizes For Moving and Non-Moving Stimuli")
    plt.hist(neuronData_0_counts,  list(range(50)), alpha = 0.5, label = "Non Moving Stimulus")
    plt.hist(neuronData_1_counts,  list(range(50)) , alpha = 0.5, label = "Moving Stimulus")
    plt.xlabel("Spikes recorded in trial", fontsize = 18)
    plt.ylabel("Spike frequency density", fontsize = 18)
    plt.legend(loc='upper right')
    plt.show()
    # plt.hist(binIntervals_0, len(neuronData_0_counts), weights = neuronData_0_counts, label= "Histogram of neuronData_0 spike counts")
    #  plt.show()

        # plot single histograms
    #   plt.bar(list(range(0, len(neuronData_0_counts))), neuronData_0_counts)
    #   plt.show()
    #   plt.bar(list(range(0, len(neuronData_1_counts))), neuronData_1_counts)
    #   plt.show()

    ## now we need to compute d'

    neuronData_0_counts = countBins(sortIntoBinsWithId(neuronData_0, 1000, 1000, trialData, 0))
    neuronData_1_counts = countBins(sortIntoBinsWithId(neuronData_1, 1000, 1000, trialData, 1))


    
    deePrime = calculateDeePrimeBinned(neuronData_0_counts, neuronData_1_counts)
    print("the d' value between the datasets of neuron A is: ", deePrime)
    
    thresholdDifference = 0.02

    optimalBoundary, predictionRate = plotOptimalBoundaryGraph(neuronData_0_counts, neuronData_1_counts)
    print("The optimal boundary of neuron A's decoder is: ", optimalBoundary)

    print("optimal prediction rate: ", predictionRate)


if(q3 == True):
    # load neuron_B_data
    trialData = np.loadtxt('trial_ID.csv', delimiter=',')
    neuronDataA = np.loadtxt('neuron_A.csv', delimiter=',')
    neuronDataB = np.loadtxt('neuron_B.csv', delimiter=',')

    # sort data into bins
    binsize = 1000
    neuronDataBinsB = sortIntoBins(neuronDataB, binsize)
    neuronDataBinsA = sortIntoBins(neuronDataA, binsize)
    # split bins by ID 0 and 1
    trialData = np.loadtxt('trial_ID.csv', delimiter=',')
    neuronDataBinsB_0, neuronDataBinsB_1 = splitDataByTrialId(neuronDataBinsB, trialData)
    neuronDataBinsA_0, neuronDataBinsA_1 = splitDataByTrialId(neuronDataBinsA, trialData)
    # now de-bin the neuron data
    neuronDataB_0 = flattenList(neuronDataBinsB_0)
    neuronDataB_1 = flattenList(neuronDataBinsB_1)

    neuronDataA_0 = flattenList(neuronDataBinsA_0)
    neuronDataA_1 = flattenList(neuronDataBinsA_1)

    # count neuronDataA and neuronDataB into bins
    neuronDataA_counts = countIntoBins(neuronDataA, 1000)
    neuronDataB_counts = countIntoBins(neuronDataB, 1000)

   # neuronDataB_counts_0 = countIntoBins(neuronDataB_0, 1000)
   # neuronDataB_counts_1 = countIntoBins(neuronDataB_1, 1000)
    
  #  neuronDataB_0_counts = countIntoBinsNoZero(neuronDataB_0, 1000)
  #  neuronDataB_1_counts = countIntoBinsNoZero(neuronDataB_1, 1000)
    neuronDataB_0_counts = countBins(sortIntoBinsWithId(np.array(neuronDataB_0), 1000, 1000, trialData, 0))
    neuronDataB_1_counts = countBins(sortIntoBinsWithId(np.array(neuronDataB_1), 1000, 1000, trialData, 1))

    neuronDataA_0_counts = countBins(sortIntoBinsWithId(np.array(neuronDataA_0), 1000, 1000, trialData, 0))
    neuronDataA_1_counts = countBins(sortIntoBinsWithId(np.array(neuronDataA_1), 1000, 1000, trialData, 1))
    
    
    plt.hist(neuronDataB_0_counts, list(range(50)), alpha = 0.5, label = "Non Moving Stimulus")
    plt.hist(neuronDataB_1_counts, list(range(50)) , alpha = 0.5, label = "Moving Stimulus")
    plt.title("Histogram of Trial Spike Count Sizes For Moving and Non-Moving Stimuli in Neuron B")
    plt.xlabel("Spikes recorded in trial", fontsize = 18)
    plt.ylabel("Spike frequency density", fontsize = 18)
    plt.legend(loc='upper right')
    plt.show()

    # compute d' value between neuron a and neuron b
    #deePrime = calculateDeePrimeBinned(neuronDataA_counts, neuronDataB_counts)
    #print("the d' value between the datasets of neuron A and neuron B is: ", deePrime)
    print("neuronDataB_0_counts + neuronDataB_1_counts: ", np.sum(neuronDataB_0_counts) + np.sum(neuronDataB_1_counts))
    deePrimeSelf = calculateDeePrimeBinned(neuronDataB_0_counts, neuronDataB_1_counts)
    print("the d' value between trial_0 and trial_1 of neuron B is: ", deePrimeSelf)
    # compute true positive rate

    # compute true negative rate

    # plot
    optimalBoundary, predictionRate = plotOptimalBoundaryGraph(neuronDataB_0_counts, neuronDataB_1_counts)
    print("The optimal boundary of neuron B's decoder is: ", optimalBoundary)
    print("optimal prediction rate: ", predictionRate)
    
    # decode stimulus from joint activity of neurons
    # Xa and Xb correspond to single-trial counts from A and B respectively
    
    
    predictedTrialIds = []

    for countA, countB in zip(neuronDataA_counts, neuronDataB_counts):
        prediction = decodeSingleTrialSpikePair(countA, countB)
        predictedTrialIds.append(prediction)
    
    # now we need to find the correct rate of our predictions by
    # comparing it against the list of trial ids
    correctList = []
    for guessID, realID in zip(predictedTrialIds, trialData):
        if(guessID == realID):
            correctList.append(1)
        else:
            correctList.append(0)
    
    correctRateIDs = np.mean(correctList)
    print("correct rate of id prediction for paired predictions: ", correctRateIDs)

    # let's plot some histograms of neuron B's spike counts for some extra marks 

    # plot both histograms
    plt.title("Spike Count for Non-Moving Stimulus per Trial in Neuron B", fontsize = 22)
    plt.bar(list(range(0, len(neuronDataB_0_counts))), neuronDataB_0_counts)
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Spike Count in Trial', fontsize=18)
    #plt.ylim(0,50)
    plt.show()

    plt.title("Spike Count for Moving Stimulus per Trial in Neuron B", fontsize = 22)
    plt.bar(list(range(0, len(neuronDataB_1_counts))), neuronDataB_1_counts)
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Spike Count in Trial', fontsize=18)
    plt.show()
    # output mean spike count of neuron B
    print(np.mean(neuronDataB_0_counts), "neuronDataB_0_counts mean")
    print(np.mean(neuronDataB_1_counts), "neuronDataB_1_counts mean")

    print(np.mean(neuronDataA_0_counts), "neuronDataA_0_counts mean")
    print(np.mean(neuronDataA_1_counts), "neuronDataA_1_counts mean")

    print(np.mean(neuronDataA_counts), "neuronDataA_counts mean")
    print(np.mean(neuronDataB_counts), "neuronDataB_counts mean")


    print(len(neuronDataA_0_counts))
    print(len(neuronDataB_0_counts))
    # try plotting difference in spikes between 0 in b and 0 in a
    differences = []

   
    for a, b in zip(neuronDataA_counts, neuronDataB_counts):
        differences.append((a)-b)

    plt.title("Spike difference in spike count for moving stimulus in A and B", fontsize = 22)
    plt.bar(list(range(0, len(neuronDataA_counts))), differences)
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Spike Count in Trial', fontsize=18)

   # plt.ylim(0,50)
    plt.show()





