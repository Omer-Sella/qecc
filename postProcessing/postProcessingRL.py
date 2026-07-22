import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass #, field
import fnmatch
import json
from qecc.polynomialCodes import generateBicycleCode
from qecc.codeUtilities import codeWeights, sameCode
#import copy 
import bisect
from openpyxl import load_workbook
from operator import attrgetter


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


# The published Bivariate Bicycle codes (Bravyi et al 2024, Table 3), keyed by (l, m),
# used as reference anchors. Polynomials as monomial-exponent lists, matching
# generateABmatrices(l, m, aX, aY, bX, bY).
REFERENCE_CODES = {
    (6, 6):  dict(name="[[72,12,6]]",   aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
    (15, 3): dict(name="[[90,8,10]]",   aX=[9],    aY=[1, 2],  bX=[0, 2],  bY=[7]),
    (9, 6):  dict(name="[[108,8,10]]",  aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
    (12, 6): dict(name="[[144,12,12]]", aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
    #(15, 3): dict(name="[[144,12,12]]", aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
    #(21, 18): dict(name="[[144,12,12]]", aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
}



@dataclass
class VisitedCode:
    l: int
    m: int
    aX: np.ndarray
    bX: np.ndarray
    aY: np.ndarray
    bY: np.ndarray
    weightsX: np.ndarray
    weightsZ: np.ndarray
    Hx: np.ndarray 
    Hz: np.ndarray 
    reward: float
    

def findWorstCodes(pathToCodeLogs, numberOfWorstCodesToGet = 100):
    aggregatedCodes = []  # bits tuple -> [counts (5,), samples, k]
    worstRewards = []
    for dirpath, _dirnames, filenames in os.walk(pathToCodeLogs):
        for name in fnmatch.filter(filenames, "codeEvaluations_*.jsonl"):
            with open(os.path.join(dirpath, name)) as fid:
                for line in fid:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        aX = record["aX"]
                        bX = record["bX"]
                        aY = record["aY"]
                        bY = record["bY"]
                        l = int(record["env_l"])
                        m = int(record["env_m"])
                        reward = float(record["reward"])
                        combinedErrorRate = np.asarray(record["logicalErrorCounts"], dtype=np.int64) + \
                                    np.asarray(record["decoderFailureCounts"], dtype=np.int64)
                        logicalQubits = int(record["numberOfLogicalQubits"])
                        Hx,Hz = generateBicycleCode(l,m, aX, aY, bX, bY)
                        wX = codeWeights(Hx)
                        wZ = codeWeights(Hz)
                        code = VisitedCode(l,m,aX,bX,aY,bY,wX, wZ, Hx, Hz, reward)
                        # Check if the new code is the same as a previously logged code:
                        if len(aggregatedCodes) < numberOfWorstCodesToGet: # Populate the list
                            bisect.insort(aggregatedCodes, code, key=attrgetter("reward"))
                            
                            
                        else: # The list is full of codes, check if this code is different
                            if not np.any([sameCode([code.Hx,code.Hz],[c.Hx,c.Hz]) for c in aggregatedCodes]): # The code is different than all codes in the list, check if its a worse performing code
                                if code.reward < aggregatedCodes[-1].reward:
                                    bisect.insort(aggregatedCodes, code, key=attrgetter("reward"))
                        if len(aggregatedCodes) > numberOfWorstCodesToGet:
                            aggregatedCodes = aggregatedCodes[0:numberOfWorstCodesToGet]
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated last line from a killed run
    return aggregatedCodes
                    
                                
                                
        
    
    
    
    

# def policyEvaluation(pathToCrawl, worstCodesForReset):   
#     """
#     At this point we have some policies, and we want to evaluate them.
#     We also have some saved codes which were logged.
#     This postprocessing script does the following:
#         a.	crawls over the save policy data and locates pth files.
#         b.	loads saved policy from pth, and 
#         c.	deterministically evaluates the policy for ~ 2 * (l + m ) iterations. 
#         d.	Optional: Resets the environment to something other than 0 and repeats (say 100 times).
#         e.	Change the decoder type and rollout again.

#     """
#     # Crawl (walk) through the folder 
#     for dirpath, _dirnames, filenames in os.walk(pathToCrawl):
#         if "policy_weights.pth" not in filenames or "experiment.txt" not in filenames:
#             # skip if you can't find saved weights or an experiment log.
#             continue
        
#         policyWeightsPath = os.path.join(dirpath, "policy_weights.pth")
#         experimentPath = os.path.join(dirpath, "experiment.txt")
#         # First get the comments from the log:
#         comments = readComments(experimentPath)
#         useDict = comments.get("env_use_dict_observation", "False").lower() == "true"
#         if not useDict:
#             continue
#         l = int(comments["env_l"])
#         m = int(comments["env_m"])



    
#     l, m = run["l"], run["m"]
#     bitFlipping = run["comments"].get("env_bit_flipping", "False").lower() == "true"
#     net, kMin = buildPolicyNet(run, qeccDataRoot)
#     maxSteps = rolloutMultiplier * max(l, m)
#     visited = []

#     env = makeEnv(l, m, kMin, bitFlipping, numberOfSamples=1,
#                   numberOfIterations=numberOfIterations, errorRange=FIVE_POINT_GRID)
#     obs, _ = env.reset(seed=0)
#     zeroVisited, zeroCycle = rollout(net, env, l, m, maxSteps, obs)
#     for step, state in enumerate(zeroVisited):
#         aX, bX, aY, bY = state
#         visited.append(VisitedCode(l, m, np.array(aX), np.array(bX), np.array(aY),
#                                    np.array(bY), "zero", step, run["name"]))

#     try:
#         worst = worstLoggedCodes(qeccDataRoot, l, m, kMin, numWorst) if numWorst > 0 else []
#     except (ValueError, FileNotFoundError) as error:
#         print(f"  item (d) skipped for {run['name']}: no usable logged codes ({error})")
#         worst = []
#     for worstIndex, (aX, bX, aY, bY, _reward) in enumerate(worst):
#         env.reset(seed=worstIndex + 1)
#         startObs = seedEnvWithCode(env, aX, bX, aY, bY)
#         worstVisited, _cycle = rollout(net, env, l, m, maxSteps, startObs)
#         for step, state in enumerate(worstVisited):
#             vaX, vbX, vaY, vbY = state
#             visited.append(VisitedCode(l, m, np.array(vaX), np.array(vbX), np.array(vaY),
#                                        np.array(vbY), f"worst-{worstIndex}", step, run["name"]))
#     return visited, zeroCycle






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
def buildExcelRowFromComments(comments, folderPath):
    """Map a run's comments + folder path to {columnLetter: value}, matching the hand-built
    parameterSweep.xlsx table (data area columns H..W)."""
    def g(key):
        v = comments.get(key)
        return "" if v in (None, "None") else v

    surrogate = comments.get("model_surrogate_model_path")
    hasSurrogate = surrogate not in (None, "None", "")
    encoderShort = ""
    if hasSurrogate:
        parts = surrogate.replace("\\", "/").rstrip("/").split("/")
        encoderShort = "/".join(p for p in parts[-2:] if p)

    code = f"{comments['env_l']},{comments['env_m']}" if "env_l" in comments and "env_m" in comments else ""

    dictForRow = {
        "H": folderPath,                              # Job name / folder name
        "I": int(comments.get("seed_for_environment")) if comments.get("seed_for_environment") else None,               # Seed
        "J": code,                                    # Code
        "K": int(comments.get("env_minimum_number_of_qubits")) if comments.get("env_minimum_number_of_qubits") else None,       # Number of qubits
        "M": comments.get("model_architecture"),                 # Network architecture
        "N": "Yes" if hasSurrogate else "No",         # Use pretrained encoder
        "O": float(comments.get("encoder_lr_factor")) if comments.get("encoder_lr_factor") else None,                  # Encoder lr factor
        "P": float(comments.get("index_to_unfreeze_encoder_updates")) if comments.get("index_to_unfreeze_encoder_updates") else None,  # Freeze until ?
        "Q": encoderShort,                            # Encoder
        "R": float(comments.get("entropy_eps")) if comments.get("entropy_eps") else None,                        # Entropy eps
        "S": "Done",                                  # Running / processed
        # T (Is good?) and U (Comments) are left blank for you to fill by hand.
    }
    return dictForRow


def appendToParameterSweep(xlsxPath, comments, folderPath, pngPath, experimentPath, sheetName="allData"):
    """Append one run as a new row to the parameter-sweep workbook. Dedupes by folder path
    (column H); adds the V/W hyperlink columns if missing. Returns True if a row was written."""
    if not xlsxPath or not os.path.exists(xlsxPath):
        print(f"[sweep] workbook not found, skipping: {xlsxPath}")
        raise ValueError("Workbook not found")
    XL_FOLDER_COLUMN = 8
    XL_COMMENTS_COLUMN = 23
    workBook = load_workbook(xlsxPath)    
    workSheet = workBook[sheetName] if sheetName else workBook.active

    if workSheet["V1"].value in (None, ""):
        workSheet["V1"] = "Plot"
    if workSheet["W1"].value in (None, ""):
        workSheet["W1"] = "Data"

    # Check if the folder already appears in the excel
    normFolder = os.path.normcase(os.path.abspath(folderPath))
    for r in range(2, workSheet.max_row + 1):
        h = workSheet.cell(row=r, column=XL_FOLDER_COLUMN).value
        if h and os.path.normcase(os.path.abspath(str(h))) == normFolder:
            print(f"[sweep] already logged (row {r}), skipping: {folderPath}")
            return False

    # Get number of last row
    last = 1
    for r in range(2, workSheet.max_row + 1):
        for col in range(XL_FOLDER_COLUMN, XL_COMMENTS_COLUMN + 1):
            if workSheet.cell(row=r, column=col).value not in (None, ""):
                last = r
                break

    target = last + 1
    for colLetter, value in buildExcelRowFromComments(comments, folderPath).items():
        workSheet[f"{colLetter}{target}"] = value

    if pngPath and os.path.exists(pngPath):
        workSheet[f"V{target}"] = f'=HYPERLINK("{os.path.abspath(pngPath)}","plot")'
    if experimentPath and os.path.exists(experimentPath):
        workSheet[f"W{target}"] = f'=HYPERLINK("{os.path.abspath(experimentPath)}","experiment.txt")'
    try:
        workBook.save(xlsxPath)
    except PermissionError:
        alt = xlsxPath.replace(".xlsx", "_autofill.xlsx")
        workBook.save(alt)
        print(f"[sweep] '{os.path.basename(xlsxPath)}' is open in Excel; wrote to "
              f"'{os.path.basename(alt)}' instead. Close Excel to append directly.")
        return True
    print(f"[sweep] appended row {target}: {folderPath}")
    return True

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

    try:
        appendToParameterSweep(
            os.path.join(os.environ.get("QECC_DATA"), "parameterSweep.xlsx"),
            comments,
            folderPath=pathBreakdown[0],
            pngPath=imageName,
            experimentPath=filePath,
        )
    except Exception as exc:
        print(f"[sweep] could not update spreadsheet: {exc}")
    
    return df




if __name__ == "__main__":
    analyseEvaluation(r"C:\Users\Omer\rl-qecc-data\2026-07-14_08-27-56\experiment.txt")


                          