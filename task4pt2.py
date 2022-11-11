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
import decimal
Vrest = -70*10**-3
Vth = -50*10**-3
Vreset = -60*10**-3
VrandomStart = random.uniform(Vrest, Vreset)

Rm = 100*10**6


Ie = 0.21*(10**-9)

timestep = 0.1*10**-3


lastVmValues = []

# call this from 0 to n
# fills in our memoized array

def modelMembrane(t0, n, dt, Iext, tm1):
    if(t0 + n*dt == 0):
        lastVmValues.append(VrandomStart)
        return VrandomStart

    else:
        vm = lastVmValues[n-1] + (dt/tm1)*(-(lastVmValues[n-1] - Vrest) +Rm*Iext)

        if(lastVmValues[n-1] >= Vth):
            print("SPIKE")
            vm = Vreset
            lastVmValues.append(vm)
            return vm
        else:
            lastVmValues.append(vm)
            return vm
# returns 0 or 1 depending on if there was a spike
def modelMembraneReturnSpike(t0, n, dt, Iext, tm1):
    if(t0 + n*dt == 0):
        lastVmValues.append(VrandomStart)
        return 0

    else:
        vm = lastVmValues[n-1] + (dt/tm1)*(-(lastVmValues[n-1] - Vrest) +Rm*Iext)

        if(vm >= Vth):
            vm = Vreset
            lastVmValues.append(vm)
            return 1
        else:
            lastVmValues.append(vm)
            return 0

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)
        
def getSynapticCurrent(r1, r2, iterations, dt, Ts):
    currentQuantumns = []
    synapseStates = np.zeros((100))
    # run for one second
    for i in range(iterations):

        for n1_j in range(50):
            # calculate if a neuron in n1_j is firing
            x = random.uniform(0, 1)
            if(x < r1*dt):
                synapseStates[n1_j] += 1
        
        
        for n2_j in range(50, 100):
            x = random.uniform(0, 1)
            if(x < r2*dt):
                synapseStates[n2_j] += 1
        # now get the total current:
        I = 0
        # foreach synapse state, calculate current
        for i in range(len(synapseStates)):
            # if the synapse state is <0, make it zero, otherwise decrease it
            synapseStates[i] -= ((synapseStates[i]) * dt)/Ts
            if(synapseStates[i] < 0):
                synapseStates[i] = 0

        totalSpikes = np.sum(synapseStates)

        # I HAVE NO IDEA WHAT V IS, I SHOULD CLARIFY THIS
        # RIGHT NOW I AM JUST GUESSING
        I = totalSpikes * gi * 40*10**-3
        currentQuantumns.append(I)        
    return currentQuantumns
       
    
q1 = False
q2 = True
# model a membrane for multiple values of n*timestep
n = 0
rv = Vrest
vms = []
timeSteps = []
#inputCurrents = [0, 0.1*10**-9, 0.2*10**-9, 0.3*10**-9, 0.4*10**-9, 0.5*10**-9]
if(q1):
    tm = 30*10**-3
    inputCurrents = list(float_range(0, 0.51*10**-9, 0.01*10**-9))

    # voltage as a function of time
    for i in (range(10000)):
        Vm = modelMembrane(0, i, timestep, Ie, tm)
        vms.append(Vm)
        timeSteps.append(i*timestep)


    plt.plot(timeSteps,vms, label = "Neuron model voltage")

    plt.legend(loc='lower right')
    plt.title("LIF Neuron Model Potential as a Function of Time")
    plt.xlabel('Time elapsed / (s)', fontsize=16)
    plt.ylabel('Neuron Potential / (V)', fontsize=14)
    plt.show()

    vms = []
    lastVmValues = []

    for i in (range(10000)):
        Vm = modelMembrane(0, i, timestep, Ie, tm)
        vms.append(Vm)
        timeSteps.append(i*timestep)

    # now do frequency stuff with varying currents
    j = 0
    spikeFrequencies = []
    iterations = 100000
    for inputCurrent in inputCurrents:
        vms.clear()
        lastVmValues.clear()
        spikeCount = 0
        for i in (range(iterations)):
            isSpike = modelMembraneReturnSpike(0, i, timestep, inputCurrent, tm)
            if(isSpike == 1):
                spikeCount += 1
        spikeFrequencies.append(spikeCount/(iterations*timestep))


    inputCurrents = [x * 10**9 for x in inputCurrents]
    plt.plot(inputCurrents,spikeFrequencies, label = "Neuron model spike frequency")

    plt.legend(loc='lower right')
    plt.title("Spike Frequency as a Function of Input Current")
    plt.xlabel('Constant input current / (nA)', fontsize=16)
    plt.ylabel('Spike Frequency / (Hz)', fontsize=14)
    plt.show()
            
    # find analytical solution later
if(q2):
    # N = 100 incoming synapses

    # split into two groups
    # N1 = 50, N2 = 50

    # N1 and N2 have different firing rates

    # model synaptic responses to spike trains
    tm = 30*10**-3
    Ts = 2*10**-3 # decay time constant
    gi = 0.5*10**-9 #Siemens
    Es = 0
    s = 1 #timewindow of s

    r1 = 10 #hz
    r2 = 100
    
    dt = 0.1*10**-3


    # find V of a synapse

    
    # get number of n1 synapses that are spiking
    quantizedInputCurrents = []
    # time in seconds / dt
    iterations = 1/dt
    iterations = int(iterations)

    # define array of active synapses

    # 0 = ready to fire
    # 0> = firing
    quantizedInputCurrents = getSynapticCurrent(r1, r2, iterations, dt, Ts)
   # print(quantizedInputCurrents)
    # now that we have a function of current, let's run the synapse model
    j = 0
    vms.clear()
    timeSteps = []
    lastVmValues = []
    for i in (range(iterations)):
        vm = modelMembrane(0, i, dt, quantizedInputCurrents[i],tm)
        vms.append(vm)
        timeSteps.append(i * dt)


    plt.plot(timeSteps, lastVmValues, label = "Neuron model potential")
    plt.legend(loc='lower right')
    plt.title("LIF Model Potential as a Function of Time With Probabalistic Synapse Input Current")
    plt.xlabel('Time Elapsed / (s)', fontsize=16)
    plt.ylabel('Neuron Potential / (V)', fontsize=14)
    plt.show()
    
    plt.show()


    ### plot graph of output spike frequency as a function of input spike freqency
    
    # case 1: r1: 0Hz -> 150Hz,   r2 = 100Hz
    # case 2: r1 = 10Hz,  r2: 0 Hz -> 150Hz

    # firstly, generate a set of values from 0 to 150
    inputRates = list(range(151))
    print(inputRates)

    iterations = 10/dt
    iterations = int(iterations)

    # for r1 varying, get a set of spike frequencies
    outputSpikeRatesTrial1 = []
    outputSpikeRatesTrial2 = []
    for inputRate in tqdm(inputRates):
        # model the neuron r1 with the current rate, while keeping r2 constant
        ## GENERATE OUR INPUT CURRENT ARAY FOR THE FIRST TRIAL
        ## -----------------------------------------------------------------
       # print(inputRate)
        quantizedInputCurrents = getSynapticCurrent(inputRate, r2, iterations, dt, Ts)

        ## now get a spike value for outputSpikeRatesTrial1
        spikeCount_trial1 = 0
        vms.clear()
        lastVmValues = []
        for i in (range(iterations)):
            isSpike = modelMembraneReturnSpike(0, i, dt, quantizedInputCurrents[i], Ts)
            if(isSpike == 1):
                spikeCount_trial1 += 1
        outputSpikeRatesTrial1.append(spikeCount_trial1/(dt*iterations))


        ## now do the same thing all over again
        ## -----------------------------------------------------------------
        quantizedInputCurrents = getSynapticCurrent(r1, inputRate, iterations, dt, Ts)
        ## now get a spike value for outputSpikeRatesTrial2
        spikeCount_trial2 = 0
        vms.clear()
        lastVmValues = []
            
        for i in (range(iterations)):
            isSpike = modelMembraneReturnSpike(0, i, dt, quantizedInputCurrents[i], Ts)
            if(isSpike == 1):
                spikeCount_trial2 += 1

        outputSpikeRatesTrial2.append(spikeCount_trial2/(dt*iterations))
plt.title("Neuron Spike Frequency as a Function of Incoming Synapse Spike Rate")
plt.plot(inputRates, outputSpikeRatesTrial1, label = "Fixed R2: 100Hz", c = "blue")
plt.plot(inputRates, outputSpikeRatesTrial2, label = "Fixed R1: 10Hz", c = "red")
plt.show()








    
    
#print(vms)


    
    







