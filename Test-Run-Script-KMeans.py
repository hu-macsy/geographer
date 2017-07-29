from subprocess import call
from submitFileWrapper import *

import os
import math
import random

iterations = 7
dimension = 2

dirString = os.path.expanduser("~/WAVE/Giesse-Repart/mesh-sequences")

for i in range(iterations):
        graphNumber = i
        formatString = '%02d' % graphNumber
        p = 16*2**graphNumber
        filename = os.path.join(dirString, "refinedtrace-"+'000'+formatString+".graph")
        n = 4500000*2**graphNumber

        minBorderNodes = 1000
        multiLevelRounds = 12
        stopAfterNoGainRounds = 200
        minGainForNextGlobalRound = 100
        gainOverBalance = 0
        skipNoGainColors = 0
        useGeometricTieBreaking = 1

        if not os.path.exists(filename):
                print(filename + " does not exist.")
        else:
            others = " --minBorderNodes="+str(minBorderNodes)
            others += " --stopAfterNoGainRounds="+str(stopAfterNoGainRounds)
            others += " --minGainForNextGlobalRound="+str(minGainForNextGlobalRound)
            others += " --useGeometricTieBreaking="+str(useGeometricTieBreaking)            
            others += " --multiLevelRounds="+str(multiLevelRounds)
            others += " --initialPartition 3"
            others += " --dimensions 2"
            commandString = assembleCommandString("parco", filename, p, others)
            submitfile = createMOABSubmitFile("msub-"+str(p)+"-"+str(i)+".cmd", commandString, "00:25:00", p, "4000mb")
            call(["msub", submitfile])

                

