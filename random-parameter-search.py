from subprocess import call
import os
import math
import random

iterations = 50
dimension = 2
minExp = -1.5
maxExp = 1

dirString = os.path.expanduser("~/WAVE/Giesse-Repart/mesh-sequences")

graphNumber = 5
formatString = '%02d' % graphNumber
p = 16*2**graphNumber
filename = os.path.join(dirString, "refinedtrace-"+'000'+formatString+".graph")
n = 4500000*2**graphNumber

scalingFactor = math.pow(float(n / p), float(1)/dimension)

for i in range(iterations):
	borderDepth = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
	stopAfterNoGainRounds = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
	minGainForNextGlobalRound = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
	gainOverBalance = random.randint(0,1)
	skipNoGainColors = random.randint(0,1)
	tieBreakingStrategy = random.randint(0,2)
	useDiffusionTieBreaking = int(tieBreakingStrategy == 1)
	useGeometricTieBreaking = int(tieBreakingStrategy == 2)
	multiLevelRounds = random.randint(0,5)

	if not useDiffusionTieBreaking:
		useGeometricTieBreaking = random.randint(0,1)

	if not os.path.exists(filename):
		print(filename + " does not exist.")
	else:
		call(["job_submit", "-p", str(p), "-c", "p", "-t", str(10), "-m", str(int(4000)), "-N", "c", "submitscriptpass", "--graphFile="+filename, "--borderDepth="+str(borderDepth), "--stopAfterNoGainRounds="+str(stopAfterNoGainRounds), "--minGainForNextGlobalRound="+str(minGainForNextGlobalRound), "--gainOverBalance="+str(gainOverBalance), "--skipNoGainColors="+str(skipNoGainColors), "--useDiffusionTieBreaking="+str(useDiffusionTieBreaking), "--useGeometricTieBreaking="+str(useGeometricTieBreaking), "--multiLevelRounds="+str(multiLevelRounds), "--dimensions="+str(dimension)])
