import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

FONT_SIZE = 12
TICKS_FONT_SIZE = 10
SUMMARY_FONT_SIZE = 10
FIGURE_TITLE_FONT_SIZE = 14
# baseline reward, not normalized to error range width, for errorRange = np.linspace(10**-4, 10**-1, 10)
baselines = {(6,6): 0.033189,
             (9,6): 0.040959,
             (15,3): 0.04218,
             (12,6): 0.038739,
             (12,12): 0.0414,
             }

TEST_ERROR_RANGE = np.linspace(10**-4, 10**-1, 10)
normalizingFactor = np.abs(np.min(TEST_ERROR_RANGE) - np.max(TEST_ERROR_RANGE))

baselinesNormalized = {key: value/normalizingFactor for key,value in baselines.items()}

def clipLine(s, maxLen=44):
        return s if len(s) <= maxLen else s[:maxLen - 3] + "..."

def readComments(filePath):
    """Parse the '# key = value' comment header of an experiment.txt into a dict of strings.
    Returns {} if the file has no comment header (older runs have none)."""
    comments = {}
    with open(filePath) as fh:
        for line in fh:
            if not line.startswith("#"):
                continue                        # data rows have no leading '#'
            body = line.lstrip("#").strip()     # "# env_l = 9" -> "env_l = 9"
            if "=" in body:
                key, _, value = body.partition("=")
                comments[key.strip()] = value.strip()   # {"env_l": "9", ...}
    return comments

def markUnfreeze(a, xData, yData, label="encoder unfrozen", boxAt=(0.5, 0.88)):
    a.annotate(
        label,
        xy=(xData, yData), xycoords="data",              # tip: the real data point at the unfreeze eval
        xytext=boxAt,      textcoords="axes fraction",   # box: a fixed clear spot (tweak per panel)
        ha="center", va="center", fontsize=FONT_SIZE, color="#7a3200",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff3e0", ec="#e08a3c", lw=1.2),
        arrowprops=dict(arrowstyle="-|>", color="#e08a3c", lw=1.3, shrinkB=6),
    )

def analyseEvaluation(filePath, baseline = None):
    
    comments = readComments(filePath)
    env_l = int(comments["env_l"]) if "env_l" in comments else None
    env_m = int(comments["env_m"]) if "env_m" in comments else None
    codeParameters = f" for code parameters (l={comments['env_l']}, m={comments['env_m']})" if "env_l" in comments else ""
    sns.set_theme()
    df = pd.read_csv(filePath, sep='\t', comment = "#")
    
    
    #fig, ax = plt.subplots(1, 3, figsize = (24, 8), sharex=True)
    fig, allAx = plt.subplots(1, 4, figsize=(26, 8), sharex=True,
                              gridspec_kw={"width_ratios": [2.2, 6, 6, 6]})
    axText, ax = allAx[0], allAx[1:]  

    sns.boxplot(data=df, x="evaluation number", y="reward", ax=ax[0], showfliers=True)
    
    dfEvalNumber = df.groupby(["evaluation number"])
    #dfEvalNumber.reward.mean().plot(ax = ax[1])
    xmin = 0
    xmax = len(dfEvalNumber.groups)
    sns.lineplot(data=df, x= "evaluation number", y="reward", ax = ax[1])

    
    
    if baseline is None and (env_l,env_m) in baselinesNormalized.keys(): # Baseline was not given by the user, so # check if there is already a baseline in the dictionary at the top of this module                    
        if "env_reward_engineering" in comments:
            if comments["env_reward_engineering"].lower() == "true": # Remember that the comments are strings !
                baseline = baselinesNormalized[(env_l,env_m)]
            else:
                baseline = baselines[(env_l,env_m)]
        
    if baseline is not None: # Wow this is bad coding ! - basically, if either the user gave baseline or the baseline was found in the dict at the top of the file ...
        ax[0].hlines(baseline, xmin, xmax, label = f"Reward for baseline code ({env_l},{env_m})")
        ax[1].hlines(baseline, xmin, xmax, label = f"Reward for baseline code ({env_l},{env_m})")

    
    entropyColumns = [c for c in ("policy entropy",
                                "policy entropy aX", "policy entropy bX",
                                "policy entropy aY", "policy entropy bY")
                    if c in df.columns]

    if entropyColumns:
        meanByEval = dfEvalNumber[entropyColumns].mean()          # DataFrame: index=eval number, one col each
        for col in entropyColumns:
            isTotal = (col == "policy entropy")
            meanByEval[col].plot(
                ax=ax[2],
                label="total" if isTotal else col.replace("policy entropy ", ""),
                color="black" if isTotal else None,               # components use the colour cycle
                linewidth=2.4 if isTotal else 1.4,
                linestyle="-"  if isTotal else "--",
                zorder=3       if isTotal else 2,
            )
        # If there is information about the entropy epsilon in the comments, include it in the title of the entropy plot.
        entropyInformationString = f" Entropy epsilon = {comments['entropy_eps']}" if "entropy_eps" in comments else ""
        titleString = f"Policy entropy (total and per polynomial block)." + entropyInformationString
        ax[2].set_title(titleString, fontsize = FONT_SIZE)
        ax[2].set_xlabel("Evaluation number", fontsize = FONT_SIZE)
        ax[2].set_ylabel("Entropy", fontsize = FONT_SIZE)
        ax[2].legend(fontsize=FONT_SIZE)
    if "Encoder freeze" in df.columns:
        frozen = df["Encoder freeze"].astype(str).str.lower() == "true"   # normalise either dtype
        unfrozenRows = df.loc[~frozen, "evaluation number"]
        unfreezeEval = unfrozenRows.min() if len(unfrozenRows) else None    # None if never unfrozen in this run
        
        
        # If there is information about the encoder weights freeze / unfreeze include it as annotation in the plot.
        if unfreezeEval is not None: 
            ax[2].axvline(unfreezeEval, color="crimson", linestyle="--", linewidth=1.5)
            ax[2].text(unfreezeEval, ax[2].get_ylim()[1], "  Encoder unfreeze",
                color="crimson", va="top", ha="left", fontsize=FONT_SIZE)

            atUnfreeze = df["evaluation number"] == unfreezeEval
            rewardY  = df.loc[atUnfreeze, "reward"].mean()          # mean reward at that eval
            entropyY = df.loc[atUnfreeze, "policy entropy"].mean()  # mean total entropy at that eval
            evalOrder = sorted(df["evaluation number"].unique())
            pos = evalOrder.index(unfreezeEval)
            markUnfreeze(ax[0], pos,          rewardY)    # boxplot:  category-index x, reward y
            markUnfreeze(ax[1], unfreezeEval, rewardY)    # lineplot: value x,          reward y
            markUnfreeze(ax[2], unfreezeEval, entropyY) 


    evalOrder = sorted(df["evaluation number"].unique())
    step = max(1, round(len(evalOrder) / 10))          # aim for ~10 labels
    tickPos = list(range(0, len(evalOrder), step))     # 0, 5, 10, ...

    # Set the figure property for the plots of the undiscounted reward.  Remember that entropy gets its own treatment.
    for i in [0,1]:
        ax[i].set_title('Undiscounted reward as a function of evaluation number' + codeParameters, fontsize = FONT_SIZE)
        ax[i].set_ylabel('Reward', fontsize = FONT_SIZE)
        ax[i].set_xlabel('Evaluation number', fontsize = FONT_SIZE)
        ax[i].legend(fontsize=FONT_SIZE)

    for a in ax:
        a.set_xticks(tickPos)
        a.set_xticklabels([evalOrder[i] for i in tickPos], fontsize = TICKS_FONT_SIZE)

    
    
    pathBreakdown = os.path.split(filePath)
    fig.suptitle(f"Evaluation summary {os.path.basename(pathBreakdown[0])}\n" , fontsize = FIGURE_TITLE_FONT_SIZE)
    figureExplanatoryText = "\n".join(clipLine(f"{k} = {v}") for k, v in comments.items()) + "\n" + clipLine(os.path.basename(pathBreakdown[0]))
    axText.text(0.0, 1.0, figureExplanatoryText, transform=axText.transAxes,
                va="top", ha="left", fontsize=SUMMARY_FONT_SIZE, family="monospace", clip_on=True)
    
    #fig.text(0.5, 0.5, 'Hello, Matplotlib!', fontsize=20, color='blue', ha='center', va='center')

    # Save output as a png image in the same folder
    
    imageName = os.path.join(pathBreakdown[0],"postProcessing.png")
    plt.tight_layout()
    plt.savefig(fname = imageName)
    plt.show()
    return df




if __name__ == "__main__":
    analyseEvaluation(r"C:\Users\Omer\rl-qecc-data\2026-07-14_08-27-56\experiment.txt")


                          