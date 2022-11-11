import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------------------------------
#FUNCTIONS

#loads data from file name given
def loadData(fileName):
    data_points = np.loadtxt(fileName, delimiter=',')
    return data_points

#Calculate Coefficient of variation
#Coefficient of Variation = Standard Deviation / Mean
def CV(data):
    mean = np.mean(data)
    sd = np.std(data)
    cv = sd/mean
    print("Coefficient of Variation = ", cv)
    return cv


#Calculates the fano factor
#FF = variance of the spike count / mean spike count
def fanoFactor(data,timeBin):    
    #count number of values in each bin
    binCount = np.array(countIntoBins(data,timeBin))
    print(binCount)
    #calculate mean and varience
    mean = np.mean(binCount)
    var = np.var(binCount)
    print("mean: ", mean, "var: ", var)
    ff = var/mean
    return ff

#Counts number of spikes in each bin
def countBin(data, max):
    count = 0
    for x in data:
        if(x<max):
            count += 1
            data.remove(x)
        else:
            break
    return count, data

def countIntoBins(data,timeBin):
    #count number of spikes across the data for each time bin
    data = data.tolist()
    binCount = []
    size = len(data)
    numberOfBins = int(np.ceil(data[size-1]/timeBin))
    for i in range(numberOfBins):
        count, data = countBin(data,(i+1)*timeBin)
        binCount.append(count)
    return binCount

#Counts number of spikes in each bin
def sortBin(data, max):
    bin = []
    for x in data:
        if(x<max):
            bin.append(x)
            data.remove(x)
        else:
            break
    return bin, data
    
def sortIntoBins(data,timeBin):
    #count number of spikes across the data for each time bin
    data = data.tolist()
    binList = []
    size = len(data)
    numberOfBins = int(np.ceil(data[size-1]/timeBin))
    print(numberOfBins)
    for i in range(numberOfBins):
        bin, data = sortBin(data,(i+1)*timeBin)
        binList.append(bin)
    return binList

def FFCalculator(data):    
    #Fano Factors
    print("Fano Factors")
    print("Bin size 100: ",fanoFactor(data,100))
    print("Bin size 200: ",fanoFactor(data,200))
    print("Bin size 500: ",fanoFactor(data,500))
    print("Bin size 1000: ",fanoFactor(data,1000))

#separate bins of values into trial 0 or 1
def separateTrials(trialData,neuronData):
    #sort into bins
    sortedData = sortIntoBins(neuronA_data,1000)
    trial0 = []
    trial1 = []
    for i in range(len(trialData)):
        if trialData[i] == 0:
            trial0.append(sortedData[i])
        elif trialData[i] == 1:
            trial1.append(sortedData[i])
        else:
            print("Error, trial neither 0 nor 1")
            break
        
    return trial0, trial1

#-----------------------------------------------------------------------------------------------------
#Question 1

#Load file 'neuron_A.csv
print("Neuron A data analysis:")
neuronA_data = loadData('neuron_A.csv')   
CV(neuronA_data)
FFCalculator(neuronA_data)

#load IDs
trial_ID_data = loadData('trial_ID.csv')
#separate into 2 lists of bins containing values
trial0Data, trial1Data = separateTrials(trial_ID_data, neuronA_data)
#flatten each into a single list
trial0Data = [item for elem in trial0Data for item in elem]
trial1Data = [item for elem in trial1Data for item in elem]

trial0Data = np.array(trial0Data)
trial1Data = np.array(trial1Data)

#calculate cv and ff for each trial
print("Trial 0: ")
CV(trial0Data)
FFCalculator(trial0Data)

print("Trial 1: ")
CV(trial1Data)
FFCalculator(trial1Data)
""" 
Now load the file trial ID.csv, which contains an array of zeros and
ones. Assume that the spike times above came from a neuron recorded from
the visual cortex of a monkey during a series of sequential trials, each lasting
one second long. So trial 4 lasts from t = 3000 to t = 3999, trial 18 lasts from
t = 17000 to t = 17999, and so on. The corresponding trial IDs are given by
the values in the trial ID array. trial id = 0 corresponds to a random-dot
visual stimulus where there was no net motion. trial id = 1 corresponds to
a moving random-dot stimulus. Now separate the spike times corresponding
to trials where trial id = 0 and trials where trial id = 1. Compute the CV
and Fano factors again separately on the data corresponding to each trial
type (for Fano factor, use the same time bins as before). State the values
you find in your report. Do these values differ from what you found above
on the total data? Offer an explanation for why or why not.
"""