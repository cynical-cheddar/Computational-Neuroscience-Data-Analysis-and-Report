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
VrandomStart = random.uniform(Vrest, Vth)

Rm = 100*10**6
tm = 30*10**-3

Ie = 0.21*(10**-9)

timestep = 0.1*10**-3

print(Vrest, Vth, Vreset, Rm, tm, Ie, timestep)
lastVmValues = []

# call this from 0 to n
# fills in our memoized array

def modelMembrane(t0, n, dt, Iext):
    if(t0 + n*dt == 0):
        lastVmValues.append(VrandomStart)
        return Vrest

    else:
        vm = lastVmValues[n-1] + (dt/tm)*(-(lastVmValues[n-1] - Vrest) +Rm*Iext)

        if(vm >= Vth):
            vm = Vreset
            lastVmValues.append(vm)
            return vm
        else:
            lastVmValues.append(vm)
            return vm
# returns 0 or 1 depending on if there was a spike
def modelMembraneReturnSpike(t0, n, dt, Iext):
    if(t0 + n*dt == 0):
        lastVmValues.append(VrandomStart)
        return 0

    else:
        vm = lastVmValues[n-1] + (dt/tm)*(-(lastVmValues[n-1] - Vrest) +Rm*Iext)

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
        
        
        
       
    
q1 = True
q2 = True
# model a membrane for multiple values of n*timestep
n = 0
rv = Vrest
vms = []
timeSteps = []
#inputCurrents = [0, 0.1*10**-9, 0.2*10**-9, 0.3*10**-9, 0.4*10**-9, 0.5*10**-9]
if(q1):
    inputCurrents = list(float_range(0, 0.51*10**-9, 0.01*10**-9))

    # voltage as a function of time
    for i in (range(10000)):
        Vm = modelMembrane(0, i, timestep, Ie)
        vms.append(Vm)
        timeSteps.append(i*timestep)


    plt.plot(timeSteps,vms, label = "Neuron model voltage")

    plt.legend(loc='lower right')
    plt.title("LIF Neuron Model Potential as a Function of Time")
    plt.xlabel('Time elapsed / (s)', fontsize=16)
    plt.ylabel('Neuron Potential / (V)', fontsize=14)
    plt.show()

    for i in (range(10000)):
        Vm = modelMembrane(0, i, timestep, Ie)
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
            isSpike = modelMembraneReturnSpike(0, i, timestep, inputCurrent)
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

    Ts = 2*10**-3 # decay time constant
    gi = 0.5*10**-9 #Siemens
    Es = 0
    s = 1 #timewindow of s

    r1 = 10 #hz
    r2 = 100



    
    
#print(vms)


    
    







