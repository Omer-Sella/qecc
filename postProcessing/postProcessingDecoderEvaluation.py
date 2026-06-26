from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse    

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
    return data['errorRange'], data['errorRate'] / data['Number of samples']

def calculateReward(inputBER, outputBER): 
    # This is exactly as the plotLogicalErrorRate function, except we add the area which represents the reward. It may cause a change in the range of the y axis, making the figure look differently.    
    fig, ax = plt.subplots()
    ax.plot(inputBER, outputBER, marker='o', linestyle='-', color='b', label='Logical Error Rate + Decoder Failure Rate')
    ax.fill_between(inputBER, outputBER, 1, alpha=0.3, color='g', label='Reward Area')
    reward = trapezoid(1 - outputBER, inputBER)
    ax.axhline(y=reward, color='r', label='Reward')
    ax.set_xlabel("Physical qubit error rate")
    ax.set_ylabel("Combined logical error rate after decoding")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    plt.show()
    
    return reward


    

if __name__ == "__main__":
    #plotLogicalErrorRate("c:/users/omer/qecc/decoderComparisonData/rbp3_bb_72_12_6.npy")
    inputBER, outputBER = plotLogicalErrorRate("c:/users/omer/qecc/decoderComparisonData/dualBPOSD_72_12_6.npy")
    print(inputBER)
    print(f"Reward was calculated as {calculateReward(inputBER, outputBER)} ")
    #plotLogicalErrorRate("C:/Users/Omer/qecc/decoderComparisonData/dualBPOSD_108_8_10.npy")
    # parser = argparse.ArgumentParser(description="Post-process decoder evaluation data.")
    # parser.add_argument('-p', '--pathToData', required=True, help='Path to data (.npy file)')
    # parser.add_argument('-o', '--pathToOutput', required=True, help='Path to produce output figures')
    # args = parser.parse_args()
    # errorRange, LER = plotLogicalErrorRate(args.pathToData)
    # print(f"Reward calculated as: {calculateReward(errorRange, LER)}")