import os
import re

#class settings:
	
#	def __init__(self):

def defaultSettings():
	epsilon = 0.03
	minBorderNodes = 100
	stopAfterNoGainRounds = 20
	minGainForNextGlobalRound = 10
	multiLevelRounds = 12
	initialPartition = 3	# 0:SFC, 3:k-means, 4:ms
	initialMigration = 0	# 0:SFC, 3:k-means, 4:ms
	gainOverBalance = 0
	skipNoGainColors = 0
	tieBreakingStrategy = 1
	repeatTimes = 10	
	
	#retString = " --dimensions=" + str(dimensions)
	retString = " --minBorderNodes="+str(minBorderNodes)
	retString += " --stopAfterNoGainRounds="+str(stopAfterNoGainRounds)
	retString += " --minGainForNextGlobalRound="+str(minGainForNextGlobalRound)
	retString += " --multiLevelRounds="+str(multiLevelRounds)
	retString += " --initialPartition="+str(initialPartition)
	retString += " --initialMigration=" + str(initialMigration)
	#retString += " --fileFormat=" + str(fileFormat)
	#retString += " --coordFormat="+ str(coordFormat)
	#retString += " --coordFile=" + coordFile
	retString += " --repeatTimes="+ str(repeatTimes)
	retString += " --storeInfo"
	
	return retString

	
def getRunNumber(path):
	runsFile = os.path.join( path,".runs")
	
	with open(runsFile,"r+") as f:
		line = f.readline().split()
		prevRun = int(line[0])
		newRun = prevRun+1
		f.seek(0)
		f.write( str(newRun) )
		
	return newRun
