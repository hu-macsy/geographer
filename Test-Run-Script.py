from subprocess import call
from submitFileWrapper import *

import os
import math
import random

iterations = 2
dimension = 2
minExp = -1.5
maxExp = 1

#dirString = os.path.expanduser("~/WAVE/Giesse-Repart/mesh-sequences")
dirString = os.path.expanduser("./meshes/hugetrace")

graphNumber = 0
formatString = '%02d' % graphNumber
p = 16*2**graphNumber
filename = os.path.join(dirString, "hugetrace-"+'000'+formatString+".graph")
n = 4500000*2**graphNumber

scalingFactor = math.pow(float(n / p), float(1)/dimension)

for i in range(iterations):
        minBorderNodes = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
        stopAfterNoGainRounds = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
        minGainForNextGlobalRound = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
        gainOverBalance = random.randint(0,1)
        skipNoGainColors = random.randint(0,1)
        tieBreakingStrategy = random.randint(0,2)
        useDiffusionTieBreaking = int(tieBreakingStrategy == 1)
        useGeometricTieBreaking = int(tieBreakingStrategy == 2)
        multiLevelRounds = random.randint(0,7)
        initialPartition = 1
        coordFormat = 1

        if not os.path.exists(filename):
                print(filename + " does not exist.")
        else:
            others = " --minBorderNodes="+str(minBorderNodes)
            others += " --stopAfterNoGainRounds="+str(stopAfterNoGainRounds)
            others += " --minGainForNextGlobalRound="+str(minGainForNextGlobalRound)
            others += " --gainOverBalance="+str(gainOverBalance)
            others += " --skipNoGainColors="+str(skipNoGainColors)
            others += " --useDiffusionTieBreaking="+str(useDiffusionTieBreaking)
            others += " --useGeometricTieBreaking="+str(useGeometricTieBreaking)
            others += " --multiLevelRounds="+str(multiLevelRounds)
            others += " --initialPartition="+str(initialPartition)
            others += " --dimensions="+str(dimensions)
            others += " --coordFormat="+str(coordFormat)

            commandString = assembleCommandString("parco", filename, p, others)
            submitfile = createMOABSubmitFile("msub-"+str(p)+"-"+str(i)+".cmd", commandString, "00:10:00", p, "4000mb")
            call(["msub", submitfile])

                

