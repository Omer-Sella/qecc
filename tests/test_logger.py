from qecc.loggerForReinforcementLearning import logger
import numpy as np
def test_logger():
    status = 'OK'
    keys = ['minimum', 'maximum', 'average', 'serialNumber']
    myLogger = logger(keys)
    myLogger.logPrint("Hello world !")
    myLogger.logPrint("Hello world !", "red")
    print(myLogger.fileName)
    print(myLogger.logPath)
    print(myLogger.fullPath)
    for i in range(10):
        myLogger.keyValue('minimum', np.random.random())
        myLogger.keyValue('maximum', 15 + np.random.random())
        myLogger.keyValue('average', 20 +np.random.random())
        myLogger.keyValue('serialNumber', 90210)
        myLogger.dumpLogger()
    return status