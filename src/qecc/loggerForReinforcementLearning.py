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
    DATA_LOGGING_PATH =  "/rds/general/user/osella/home/rl-qecc-data/"
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
            date_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

            self.logPath = str(DATA_LOGGING_PATH)  + "/" + date_string + "_" + str(os.getpid())# "/%i" %int(time.time())
        else:
            self.logPath = logPath
        if os.path.exists(self.logPath):
            print("Warning: Log dir %s already exists."%self.logPath)
        else:
            print(f"Creating log dir {self.logPath}...")
            os.makedirs(self.logPath)
            print(f"Checking the directory was created: {os.path.exists(self.logPath)}")
        #self.fileName = os.path.join(self.logPath, fileName)
        self.fileName = fileName
        self.fullPath = os.path.join(self.logPath, self.fileName)
        self.hdf5FileName = os.path.join(self.logPath, hdf5FileName)
        
        self.currentRow = {}
        self.columnKeys = []
        
        for key in keys:
            self.columnKeys.append(key)
        
        self.headerWritten = False
            
        self.dataSet = 0
        
    def logPrint(self, message, colour='green'):
        #if mpiProcessID() == 0:
        print(colourString(message, colour, bold = True))
    
    def keyValue(self, key, value):
        if (key in self.columnKeys) and (not key in self.currentRow):
            self.currentRow[key] = value
        return 'OK'
        
    def dumpLogger(self, printOut = True):
        #if mpiProcessID() == 0:
        if self.headerWritten == False:
            with open(self.fullPath, 'a') as fid:
                fid.write("\t".join(self.columnKeys)+"\n")
            self.headerWritten = True
        values = []
        keyLengths = []
        for key in self.columnKeys:
            keyLengths.append(len(key))
                
        maximalKeyLength = max(15,max(keyLengths))
        keyString = '%'+'%d'%maximalKeyLength
        stringFormat = "| " + keyString + "s | %15s |"
        numberOfDashes = 22 + maximalKeyLength
        if printOut:
            print("-"*numberOfDashes)
        for key in self.columnKeys:
            value = self.currentRow.get(key, "")
            if isinstance(value, np.ndarray):
                valueString = np.array2string(value, max_line_width = UTILITY_FUNCTIONS_BIG_NUMBER, threshold = np.inf)
            elif hasattr(value, "__float__"):
                valueString = str(value)# TODO Omer: I temporarily placed this under comment, need to figure out if we want all these logits."%8.3g"%value
            else:
                valueString = value
            if printOut:
                print(stringFormat%(key, valueString))
            values.append(valueString)
        if printOut:
            print("-"*numberOfDashes, flush=True)
        
        with open(self.fullPath, 'a') as fid:
            fid.write("\t".join(map(str,values))+"\n")
            fid.flush()    
            self.currentRow.clear()
    

    def addComment(self, comment):
        # comment is assumed to be a string, each string is a comment
        with open(self.fullPath, 'a') as fid:
            fid.write("# " + comment + "\n")
            fid.flush()    
        
    def setupPytorchSave(self, parametersToSave):
        self.pytorchElementsToSave = parametersToSave



if __name__ == '__main__':
    pass