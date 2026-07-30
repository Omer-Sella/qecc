import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
from qecc.utils import calculateRewardFromSamples
import fnmatch

def plotLogicalErrorRate(filePath, baseline = None):
    data = np.load(filePath, allow_pickle=True).item()
    fig, ax = plt.subplots()
    combinedErrorRate = data['errorRate'] / data['Number of samples'] # This is an assumption, meaning that we want to treat all decoder failures as logical errors.
    
    ax.plot(data['errorRange'], combinedErrorRate, marker='o', linestyle='-', color='b', label='Logical Error Rate')
    if "Code name" in data.keys():
        ax.set_title(f"Evaluation of {data['Code name']} for {data['time']} seconds")
    else:
        ax.set_title(f"Evaluation for {data['time']} seconds")
    ax.set_xlabel("Physical qubit error rate")
    ax.set_ylabel("Combined logical error rate after decoding")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    prefix = os.path.splitext(filePath)[0]
    fig.savefig(prefix + "_plot.png")
    #plt.show()
    return data

   

if __name__ == "__main__":
    
    dataFolder = "c:/users/omer/qecc/decoderComparisonData/geometric5/"
    for dirpath, dirnames, filenames in os.walk(dataFolder):
        decoderEvaluationFiles = fnmatch.filter(filenames, "*.npy")
        
        for d in decoderEvaluationFiles:
            data = plotLogicalErrorRate(os.path.join(dirpath, d))
            reward = calculateRewardFromSamples(logicalErrorCount = data["errorRate"], numberOfSamples=data["Number of samples"], errorRange = data["errorRange"], l = 0, m = 0, rewardEngineering = True)
            
            print(f"Reward for code {data["Code name"]} was calculated as {reward} ")
