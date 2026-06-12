# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:26:10 2021

@author: Omer Sella
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
from matplotlib import animation
#from mpi_tools import proc_id as mpiProcessID

DATA_LOGGING_PATH = os.environ.get('QECC_DATA')
if DATA_LOGGING_PATH is None:
    # Try to avoid relying on this, instead define a system variable
    DATA_LOGGING_PATH = os.path.expanduser('~') + "/qeccReinforcementLearningData/"
# When logging (printing, writing to csv etc.) numpy arrays, if there are 
#"too many" elements the array will contain ellipsis look like this:
#[0, 0, 0, ..., 0, 0, 0] so when using array2string we need to set a threshold
#only overwhich np will use ellipsis
UTILITY_FUNCTIONS_BIG_NUMBER = 9000

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def numToBits(number, numberOfBits):
    assert number < 16
    assert number >= 0
    newNumber = np.zeros(numberOfBits, dtype = int)
    for j in range(numberOfBits - 1, -1, -1):
        newNumber[j] = newNumber[j] + (number % 2)
        number = number >> 1
    return newNumber

class plotter():
    
    def __init__(self, epochs):
        self.epochs = epochs
        #self.numberOfStepsPerEpoch = numberOfStepsPerEpoch
        #self.maximumEpisodeLength = maximumEpisodeLength
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 4)
        self.ax1.set_title('current episode rewards')
        self.ax1.set_ylabel('undiscounted reward')
        self.ax1.set_xlabel('time')
        #self.fig, self.axs = plt.subplots(2,2)
        #plt.ion()
        #self.axs[0,0].set_title('current episode rewards')
        #self.axs[0,0].set_ylabel('undiscounted reward')
        #self.axs[0,0].set_xlabel('time')
        #self.axs[1,0].set_title('previous episode returns')
        #self.axs[1,0].set_ylabel('return')
        #self.axs[1,0].set_xlabel('episode number')
        #self.axs[1,1].set_title('previous epochs')
        #self.axs[1,1].set_ylabel('return')
        #self.axs[1,1].set_xlabel('epoch number')
        self.currentRewards = []
        self.epochsDone = []
        self.counter = 0
        self.images = []
        #self.camera = Camera(self.fig)
        
        
    def step(self, reward, duration = None):
        self.currentRewards.append(reward)
        self.epochsDone.append(self.counter + 1)
        self.counter = self.counter + 1
        #self.axs[0,0].clear()
        #self.axs[0,0].set_title('Episode rewards')
        #self.axs[0,0].set_ylabel('undiscounted reward')
        #self.axs[0,0].set_xlabel('Epoch number')
        #self.axs[0,0].set_xticks(np.arange(len(self.currentRewards)))
        #plt.sca(self.axs[0, 0])
        #plt.xticks(range(len(self.currentRewards)), range(len(self.currentRewards)))
        #self.axs[0,0].scatter(np.arange(len(self.currentRewards)), self.currentRewards)
        #image = self.fig
        #plt.pause(0.1)
        #self.camera.snap()
        image = self.ax1.scatter(np.arange(len(self.currentRewards)), self.currentRewards)
        self.images.append([image])
        #plt.ion()
        #plt.pause(0.001)
        #plt.show()

    def saveAnimation(self, fileName):
        ani = animation.ArtistAnimation(self.fig, self.images, interval=50, blit=True, repeat_delay=1000)
        ani.save(fileName)
        plt.show()
        return 'OK'

    
        
        
        
        

def colourString(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class logger():
    
    def __init__(self, keys, logPath = None, hdf5FileName = 'experiment.h5', fileName = 'experiment.txt'):

#        if  mpiProcessID() == 0:
        if logPath == None:
            self.logPath = str(DATA_LOGGING_PATH) + "/temp/experiments/%i" %int(time.time())
        else:
            self.logPath = logPath
        if os.path.exists(self.logPath):
            print("Warning: Log dir %s already exists."%self.logPath)
        else:
            os.makedirs(self.logPath)
        #self.fileName = os.path.join(self.logPath, fileName)
        self.fileName = fileName
        self.hdf5FileName = os.path.join(self.logPath, hdf5FileName)
        # else:
        #     self.logPath = None
        #     self.fileName = None
        #     self.hdf5FileName = None
        self.currentRow = {}
        self.columnKeys = []
        # if agentSeed != None:
        #     self.environmentSeed = environmentSeed
        # if environmentSeed != None:
        #     self.agentSeed = agentSeed
        for key in keys:
            self.columnKeys.append(key)
        #if mpiProcessID() == 0:            
        with open(self.fileName, 'w') as fid:
            fid.write("\t".join(self.columnKeys)+"\n")
            #with h5py.File(self.hdf5FileName, 'a') as fid:
            #    for key in self.columnKeys:
            #        fid.create_group(key)
        self.dataSet = 0
        
    def logPrint(self, message, colour='green'):
        #if mpiProcessID() == 0:
        print(colourString(message, colour, bold = True))
    
    def keyValue(self, key, value):
        if (key in self.columnKeys) and (not key in self.currentRow):
            self.currentRow[key] = value
        return 'OK'
        
    def dumpLogger(self):
        #if mpiProcessID() == 0:
        values = []
        keyLengths = []
        for key in self.columnKeys:
            keyLengths.append(len(key))
                
        maximalKeyLength = max(15,max(keyLengths))
        keyString = '%'+'%d'%maximalKeyLength
        stringFormat = "| " + keyString + "s | %15s |"
        numberOfDashes = 22 + maximalKeyLength
        print("-"*numberOfDashes)
        for key in self.columnKeys:
            value = self.currentRow.get(key, "")
            if isinstance(value, np.ndarray):
                valueString = np.array2string(value, max_line_width = UTILITY_FUNCTIONS_BIG_NUMBER, threshold = UTILITY_FUNCTIONS_BIG_NUMBER)
            elif hasattr(value, "__float__"):
                valueString = str(value)# TODO Omer: I temporarily placed this under comment, need to figure out if we want all these logits."%8.3g"%value
            else:
                valueString = value
            print(stringFormat%(key, valueString))
            values.append(valueString)
        print("-"*numberOfDashes, flush=True)
        if self.fileName is not None:
            with open(os.path.join(self.logPath, self.fileName), 'a') as fid:
                fid.write("\t".join(map(str,values))+"\n")
                fid.flush()
            
            
            self.currentRow.clear()
                
    def setupPytorchSave(self, parametersToSave):
        self.pytorchElementsToSave = parametersToSave

def testLogger():
    status = 'OK'
    keys = ['minimum', 'maximum', 'average', 'serialNumber']
    myLogger = logger(keys)
    myLogger.logPrint("Hello world !")
    myLogger.logPrint("Hello world !", "red")
    for i in range(10):
        myLogger.keyValue('minimum', np.random.random())
        myLogger.keyValue('maximum', 15 + np.random.random())
        myLogger.keyValue('average', 20 +np.random.random())
        myLogger.keyValue('serialNumber', 90210)
        myLogger.dumpLogger()
    return status

def testPlotter():
    
    myPlotter = plotter(epochs = 50)
    for i in range(50):
        myPlotter.step(np.random.random())
    fileName = DATA_LOGGING_PATH + "/localData/plotterTest/testingPlotter.mp4"
    status = myPlotter.saveAnimation(fileName)
    return status


if __name__ == '__main__':
    print(testLogger())
    #print(status)
    #status = testPlotter()