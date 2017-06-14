from subprocess import call
from submitFileWrapper import *

import os
import math
import random

dimension = 2

#dirString = os.path.expanduser("~/WAVE/Giesse-Repart/mesh-sequences")

timeStepNumbers = [0, 10, 20, 30, 40, 500]

for timestep in timeStepNumbers:
    filename = "octree_timestep_"+str(timestep)+".dat.graph"
    multiLevelRounds = 9
    p = 16
    n = 200000

    if (timestep > 100):
        multiLevelRounds = 15
        p = 64
        n = 14000000

    scalingFactor = math.pow(float(n / p), float(1)/dimension)
 
    if not os.path.exists(filename):
        print(filename + " does not exist.")
    else:
        others = " --minBorderNodes="+str(100)
        others += " --stopAfterNoGainRounds="+str(200)
        others += " --minGainForNextGlobalRound="+str(500)
        others += " --gainOverBalance="+str(1)
        others += " --skipNoGainColors="+str(1)
        others += " --useDiffusionTieBreaking="+str(1)
        others += " --useGeometricTieBreaking="+str(0)
        others += " --multiLevelRounds="+str(multiLevelRounds)
        commandString = assembleCommandString("parco", filename, p, others)
        submitfile = createMOABSubmitFile("msub-"+str(p)+"-"+str(timestep)+".cmd", commandString, "00:10:00", p, "4000mb")
        call(["msub", submitfile])
