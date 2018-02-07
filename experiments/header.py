import os
import re
from inspect import currentframe, getframeinfo



# all avaialble tools
allTools = ["Geographer", "geoKmeans", "geoSfc", "parMetisGeom", "parMetisGraph", "parMetisSfc", "zoltanRib", "zoltanRcb", "zoltanHsfc", "zoltanMJ"]
allCompetitors = allTools[3:]
NUM_TOOLS = len( allTools)
NUM_COMPETITORS = len( allCompetitors)

# absolute paths for the executable of each tool
geoExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/Diamerisi"
initialExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/geomDiamerisi"
parMetisExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/parMetisExe"
competitorsExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/competitorsExe"
allCompetitorsExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/allCompetitorsExe"

# other paths
basisPath = os.path.expanduser("~/parco-repart/Implementation/experiments/")
competitorsPath = os.path.join( basisPath, "competitors" )
toolsPath = os.path.join( basisPath, "tools" )
plotsPath = os.path.join( basisPath, "plots" )

NUM_METRICS = 10
METRIC_NAMES = ['timeTotal', 'finalCut', 'imbalance', 'maxBnd', 'totBnd', 'maxCommVol', 'totCommVol', 'maxBndPercnt', 'avgBndPercnt', 'timeSpMV']
METRIC_VALUES = [ 'seconds', 'number of edges', 'ratio', 'number of vertices', 'number of vertices', 'number of vertices', 'number of vertices', 'ratio', 'ratio', 'seconds']

class experiment:
	def __init__(self):
		self.expType = -1	# 0 weak, 1 strong, 2 other
		self.size = 0
		self.ID = 0
		self.dimension = -1
		self.fileFormat = -1
		
		self.graphs = []
		self.paths = []
		self.k = []


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

# Parses a SaGa config file with a specific format.
# Returns a list of classes experiment containing all the experiments described in the config file
		
def parseConfigFile( configFile ):
	inPath = ""
	expTypeFlag = -1; # 0 for weak, 1 for strong
	lineCnt = 0
	expID = 0

	allExperiments = []

	with open(configFile) as f:
		line = f.readline(); lineCnt +=1
		keyword = line.split()
		if keyword[0]=="path":		# first line is the path of the input 
			inPath= keyword[1]
			
		print("path: " + inPath)	
		line = f.readline(); lineCnt +=1
		# for all the other lines in the file
		
		while line:
			expHeader = "-"
			if line[0]=='%':			# skip comments
				line = f.readline(); lineCnt +=1
				continue;
			else:
				expHeader = re.split('\W+',line)
				
			#print(expHeader)
			
			# traverse the experiment header line to store parameters
			dimension = -1
			fileFormat = -1
			for i in range(1, len(expHeader) ):
				if expHeader[i]=="dim":
					dimension = expHeader[i+1]
				elif expHeader[i]=="fileFormat":
					fileFormat = expHeader[i+1]
					if int(fileFormat)<0 or int(fileFormat)>6:
						print("ERROR: unknown file format " + str(fileFormat) + ".\nAborting...")
						exit(-1)
				
			# create and store experiments in an array
			
			# for weak scaling
			if expHeader[0]=="weak":		
				exp = experiment()
				exp.expType = 0
				exp.dimension = dimension
				exp.fileFormat = fileFormat
				exp.ID= expID
				expID += 1
				assert(int(exp.dimension)>0),"Wrong or missing dimension in line "+str(lineCnt)
				assert(int(exp.fileFormat)>=0),"Wrong or missing fileFormat in line "+str(lineCnt)
				expData = f.readline(); lineCnt +=1
				if len(expData)==0:
					print("WARNING: each experiment must end with a \#");
					
				while expData[0]!="#":
					expTokens = expData.split()
					for i in range(0, len(expTokens)-1):
						exp.graphs.append( expTokens[0] )
						exp.paths.append( os.path.join(inPath, expTokens[0]) )
						exp.k.append(expTokens[i+1])
						exp.size +=1
					expData = f.readline(); lineCnt +=1
					if len(expData)==0:
						print ("WARNING: possibly forgot \'#\' at end of an experiment")
					elif expData[0]=="weak" or expData[0]=="strong" or expData[0]=="other": 
						print ("WARNING: possibly forgot \'#\' at end of an experiment")
					
				allExperiments.append(exp)
				
			# for strong scaling
			elif expHeader[0]=="strong":	
				expData = f.readline(); lineCnt +=1
				if len(expData)==0:
					print("WARNING: each experiment must end with a \#");
					
				while expData[0]!="#":
					exp = experiment()
					exp.expType = 1
					exp.dimension = dimension
					exp.fileFormat = fileFormat
					exp.ID= expID
					expID += 1
					assert(int(exp.dimension)>0), "Wrong or missing dimension in line "+str(lineCnt)
					assert(int(exp.fileFormat)>=0),"Wrong or missing fileFormat in line "+str(lineCnt)
					expTokens = expData.split()
					
					for i in range(0, len(expTokens)-1):
						exp.graphs.append( expTokens[0] )
						exp.paths.append( os.path.join(inPath, expTokens[0]) )
						exp.k.append(expTokens[i+1])
						exp.size +=1
					allExperiments.append(exp)
					expData = f.readline(); lineCnt +=1
		
			# miscellanous experiments
			elif expHeader[0]=="other":		
				expData = f.readline(); lineCnt +=1

				while expData[0]!="#":
					exp = experiment()
					exp.expType = 2
					exp.dimension = dimension
					exp.fileFormat = fileFormat
					exp.ID= expID
					expID += 1
					assert(int(exp.dimension)>0), "Wrong or missing dimension in line "+str(lineCnt)
					assert(int(exp.fileFormat)>=0),"Wrong or missing fileFormat in line "+str(lineCnt)
					expTokens = expData.split()
					
					for i in range(0, len(expTokens)-1):
						exp.graphs.append( expTokens[0] )
						exp.paths.append( os.path.join(inPath, expTokens[0]) )
						exp.k.append(expTokens[i+1])
						exp.size +=1
					
					allExperiments.append(exp)
					expData = f.readline(); lineCnt +=1
			else:
				print("Undefined token: " + expHeader[0])

			line = f.readline(); lineCnt +=1
		#while line

	return allExperiments

#######################################################################
# parses an outFile and returns the metric name found and the actual metric values

def parseOutFile( outFile ):
	
	if not os.path.exists(outFile):
		print ("WARNING: File "+outFile+" does not exist.\nSkipping...");
		return [-1]*NUM_METRICS, [-1]*NUM_METRICS, -1
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
			#if tokens[0]=="numBlocks=":
			#	n = tokens[1]	
		
		metricNames = f.readline().split()
		#print(metricNames)
		line = f.readline()
		metricValues = [ float(x) for x in line.split()]
		
		# in case metrics are less than NUM_METRICS, fill the rest with -1
		for i in range(len(metricValues), NUM_METRICS):
			metricValues.append(-1)
			metricNames.append("-")
		#print(metricValues)
		
	return metricNames, metricValues, n
		
#######################################################################

def outFileString( exp, i, tool):
	if i>exp.size:
		print("WARNING: request for experiment run " + str(i) + " but experiment " +str(exp.ID) +" has size only " +str(exp.size) )
		return ""
	if tool not in allTools:
		print("WARNING: tool " + tool + " is invalid")
		return ""
	
	outFile = exp.graphs[i].split('.')[0] + "_k" + str(exp.k[i]) +"_"+ tool + ".info"
	
	return os.path.join( toolsPath, tool , outFile)

#######################################################################

def printWarning( msg):
	frameinfo = getframeinfo(currentframe())
	print( frameinfo.filename +", " + str(frameinfo.lineno) + ": WARNING: " + msg)
		
def printError( msg):
	frameinfo = getframeinfo(currentframe())
	print( frameinfo.filename +", " + str(frameinfo.lineno) + ": ERROR" + msg)	