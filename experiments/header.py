import os
import re


# all avaialble tools
allTools = ["Geographer", "parMetisGeom", "parMetisGraph"]

# absolute paths for the executable of each tool
geoExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/Diamerisi"
parMetisExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/parMetisExe"

# other paths
basisPath = os.path.expanduser("~/supermuc/parco-repart/Implementation/experiments/")
competitorsPath = os.path.join( basisPath, "competitors" )




class experiment:
	def __init__(self):
		self.expType = -1	# 0 weak, 1 strong, 2 other
		self.size = 0
		self.dimension = -1
		self.fileFormat = -1
		
		self.graphs = []
		self.paths = []
		self.k = []

	#def submitExp(self):
		
		

	def printExp(self):
		if self.expType==0:
			print("Experiment type: weak" )
		elif self.expType==1:
			print("Experiment type: strong" )
		elif self.expType==2:
			print("Experiment type: other" )
		else:
			print("Unrecognized type: " + str(self.expType) )

		print("Number of experiments: " + str(self.size) + ", dim= " + str(self.dimension) + ", fileFormat: " + str(self.fileFormat) )
		
		for i in range(0, self.size):
			print("graph: " + self.graphs[i] + " , k= "+ str(self.k[i]) + ",\t full path: " + self.paths[i]  )

	#def (self, i):
	#	return self.graph[i]+"_"+self.k[i]

#######################################################################

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

#######################################################################

def getRunNumber(path):
	runsFile = os.path.join( path,".runs")
	
	with open(runsFile,"r+") as f:
		line = f.readline().split()
		prevRun = int(line[0])
		newRun = prevRun+1
		f.seek(0)
		f.write( str(newRun) )
		
	return newRun

#######################################################################

def parseOutFile( outFile ):
	
	if not os.path.exists(outFile):
		print ("WARNING: File "+outFile+" does not exist.\nSkipping...");
		return [], [], -1
		#exit(-1)
	else:
		print ("Parsing outFile: " + outFile)
	n = -1
		
	with open(outFile) as f:
		line = f.readline()
		tokens = line.split()

		if len(tokens)==0:	#found an empty line
			tokens = "0"
			#print(len(tokens))
		while tokens[0]!="gather":
			line = f.readline();
			tokens = line.split()
			if len(tokens)==0:
				tokens = "0"
			#print(tokens)
			if tokens[0]=="numBlocks=":
				n = tokens[1]	
		
		metricNames = f.readline().split()
		#print(metricNames)
		line = f.readline()
		metricValues = [ float(x) for x in line.split()]
		#print(metricValues)
		
	return metricNames, metricValues, n
		
			
						
