"""
Walk a dataFolder folder and run analyseEvaluation on every '*experiment.txt' file that
lives in a subfolder which does NOT already contain a '*.png' file (i.e. runs that
have not been post-processed yet). 
Folders that already have a png are skipped.

Usage:
    python crawlEvaluations.py                      # Gets the data from an environment variable QECC_DATA
    python crawlEvaluations.py <dataFolder folder>        # crawl a different dataFolder
"""

import os
import sys
import fnmatch
# Non-interactive backend, set BEFORE importing anything that pulls in pyplot,
# so analyseEvaluation's plt.show() becomes a no-op and the crawl runs unattended.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataFolder = os.environ.get("QECC_DATA")

# Make postProcessingRL importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postProcessingRL import analyseEvaluation


def crawl(dataFolder, baseline=None):
    processed = 0
    skipped = 0
    failed = 0

    for dirpath, dirnames, filenames in os.walk(dataFolder):
        experiments = fnmatch.filter(filenames, "*experiment.txt")
        if not experiments:
            continue

        if fnmatch.filter(filenames, "*.png"):          # already post-processed
            print(f"SKIP (png exists): {dirpath}")
            skipped += 1
            continue

        for experiment in experiments:
            experimentPath = os.path.join(dirpath, experiment)
            try:
                print(f"PROCESS: {experimentPath}")
                analyseEvaluation(experimentPath, baseline=baseline)
                processed += 1
            except Exception as exc:
                print(f"FAILED:  {experimentPath}  ->  {exc}")
                failed += 1
            finally:
                plt.close("all")                        # free the figure before the next file

    print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    crawl(dataFolder)