import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import seaborn as sns
import argparse




def analyseEvaluation(filePath, baseline = None):
    
    df = pd.read_csv(filePath, sep='\t')
    keys = df.columns.values
    
    
    fig, ax = plt.subplots(1,3, figsize = (16, 16))
    
    dfEvalNumber = df.groupby(["evaluation number"])
    
    dfEvalNumber.reward.mean().plot(ax = ax[0])
    dfEvalNumber.boxplot(column = "reward", ax = ax[1], showfliers = True)
    
    
    xmin = 0
    xmax = len(dfEvalNumber.groups)
    if baseline != None: 
        ax[0].hlines(baseline, xmin, xmax)
    ax[0].set_title('Undiscounted reward as a function of evaluation number')
    ax[0].set_ylabel('Reward')
    ax[0].set_xlabel('Evaluation number')
    
    
    if "policy entropy" in df.keys():
        df["policy entropy"].plot(ax=ax[2])
        ax[2].set_title('Policy entropy')
        ax[2].set_xlabel('Evaluation number')
    
    
    
    pathBreakdown = os.path.split(filePath)
    imageName = pathBreakdown[0] + "/postProcessing.png"
    plt.tight_layout()
    plt.savefig(fname = imageName)
    plt.show()
    return df




if __name__ == "__main__":
    analyseEvaluation(r"C:\Users\Omer\qecc\rl-qecc-data\2026-07-02_09-48-20\experiment.txt")
    """
    myEvaluationKeys = ["evaluation number",
                    "observation",
                    "action",
                    "reward",
                    "policy entropy"]
    """
    

                          