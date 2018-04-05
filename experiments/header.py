import os
import re
import sys
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
repartAllExe =  "/home/hpc/pr87si/di36qin/parco-repart/Implementation/repartitionWithAll"
repartKmeans =  "/home/hpc/pr87si/di36qin/parco-repart/Implementation/repartKmeans"

# other paths
basisPath = os.path.expanduser("~/parco-repart/Implementation/experiments/")
competitorsPath = os.path.join( basisPath, "competitors" )
toolsPath = os.path.join( basisPath, "tools" )
plotsPath = os.path.join( basisPath, "plots" )

METRIC_NAMES = ['timeTotal', 'finalCut', 'imbalance', 'maxBnd', 'totBnd', 'maxCommVol', 'totCommVol', 'maxDiameter', 'avgDiameter','numDisconBlocks', 'timeSpMV', 'timeComm']
METRIC_VALUES = [ 'seconds', 'number of edges', 'ratio', 'number of vertices', 'number of vertices', 'number of vertices', 'number of vertices', 'number of vertices', 'number of vertices', 'number of blocks', 'seconds', 'seconds']
NUM_METRICS = len(METRIC_NAMES)

# global settings for all 
epsilon = 0.03


class experiment:
	def __init__(self):
		self.expType = -1	# 0 weak, 1 strong, 2 other
		self.size = 0
		self.ID = 0
		self.dimension = -1
		self.fileFormat = -1
		self.coordFormat = -1
		
		self.graphs = []
		self.paths = []
		self.coordPaths = []
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
		
		print("Number of experiments: " + str(self.size) + ", dim= " + str(self.dimension) + ", fileFormat: " + str(self.fileFormat) + ", coordFormat: " + str(self.coordFormat) )
		
		for i in range(0, self.size):
			sys.stdout.write("graph: " + self.graphs[i] + " , k= "+ str(self.k[i]) + ",\t full path: " + self.paths[i] )
			if self.coordPaths is not None and len(self.coordPaths)>0:
				if self.coordPaths[i] != "-":
					sys.stdout.write(", coordFile: " + self.coordPaths[i])
			
			sys.stdout.write("\n")

	#def (self, i):
	#	return self.graph[i]+"_"+self.k[i]

#######################################################################

def defaultSettings():
	#epsilon = 0.03
	minBorderNodes = 1000
	stopAfterNoGainRounds = 100
	minGainForNextGlobalRound = 100
	multiLevelRounds = 12
	initialPartition = 3	# 0:SFC, 3:k-means, 4:ms
	initialMigration = 0	# 0:SFC, 3:k-means, 4:ms
	gainOverBalance = 0
	skipNoGainColors = 0
	tieBreakingStrategy = 1
	repeatTimes = 5	
	
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
	#retString += " --epsilon=" + str(epsilon)
	
	return retString


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
			coordFormat = -1
			
			for i in range(1, len(expHeader) ):
				if expHeader[i]=="dim":
					dimension = expHeader[i+1]
				elif expHeader[i]=="fileFormat":
					fileFormat = expHeader[i+1]
					if int(fileFormat)<0 or int(fileFormat)>8:
						print("ERROR: unknown file format " + str(fileFormat) + ".\nAborting...")
						exit(-1)
				elif expHeader[i]=="coordFormat":
					coordFormat = expHeader[i+1]
					if int(coordFormat)<0 or int(coordFormat)>8:
						print("ERROR: unknown file format " + str(coordFormat) + ".\nAborting...")
						exit(-1)
					
			# if no coordFormat was given assume the same as the fileFormat
			#if coordFormat==-1:
			#	coordFormat = fileFormat
				
				
			# create and store experiments in an array
			
			# for weak scaling
			if expHeader[0]=="weak":		
				exp = experiment()
				exp.expType = 0
				exp.dimension = dimension
				exp.fileFormat = fileFormat
				exp.coordFormat = coordFormat
				exp.ID= expID
				expID += 1
				assert(int(exp.dimension)>0),"Wrong or missing dimension in line "+str(lineCnt)
				assert(int(exp.fileFormat)>=0),"Wrong or missing fileFormat in line "+str(lineCnt)
				expData = f.readline(); lineCnt +=1
				if len(expData)==0:
					print("WARNING: each experiment must end with a \#");
					
				while expData[0]!="#":
					expTokens = expData.split()
					#first token is the graph file
					exp.graphs.append( expTokens[0] )
					exp.paths.append( os.path.join(inPath, expTokens[0]) )
					
					#second token is the coord file, third token is k
					
					#coord file can be omitted
					if len( expTokens)==2:	#assume coord file is missing
						exp.k.append(expTokens[1])
					else:	#len(expTokens)=3
						exp.coordPaths.append( os.path.join(inPath, expTokens[1]) )
						exp.k.append(expTokens[2])
					
					exp.size +=1
						
					expData = f.readline(); lineCnt +=1
					if len(expData)==0:
						print ("WARNING: possibly forgot \'#\' at end of an experiment")
					elif expData[0]=="weak" or expData[0]=="strong" or expData[0]=="other": 
						print ("WARNING: possibly forgot \'#\' at end of an experiment")
				#exp.printExp()
				allExperiments.append(exp)
				
			# for strong scaling or miscellanous experiments
			elif expHeader[0]=="strong" or expHeader[0]=="other":	
				expData = f.readline(); lineCnt +=1
				if len(expData)==0:
					print("WARNING: each experiment must end with a \#");
					
				while expData[0]!="#":
					exp = experiment()
					if expHeader[0]=="strong":
						exp.expType = 1
					else:
						exp.expType = 2
					exp.dimension = dimension
					exp.fileFormat = fileFormat
					exp.coordFormat = coordFormat
					exp.ID= expID
					expID += 1
					assert(int(exp.dimension)>0), "Wrong or missing dimension in line "+str(lineCnt)
					assert(int(exp.fileFormat)>=0),"Wrong or missing fileFormat in line "+str(lineCnt)
					expTokens = expData.split()
					
					
					for i in range(2, len(expTokens)):
						#first token is the graph file
						exp.graphs.append( expTokens[0] )
						exp.paths.append( os.path.join(inPath, expTokens[0]) )
						#second token is the coord file
						#WARNING: coord file CANNOT be omitted
						if expTokens[1]!="-":
							exp.coordPaths.append( os.path.join(inPath, expTokens[1]) )
						else:
							exp.coordPaths.append( expTokens[1] )
						exp.k.append(expTokens[i])
						exp.size +=1
					
					#exp.printExp()
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
			if line=="":
				break
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
		print(metricValues)
		
	return metricNames, metricValues, n
		
		
#TODO: remove this function and deal with repartitio file properly
#	 (it happens because keyword "gather" appears two times in the repartition files...)
def parseRepartFile( outFile ):
	
	if not os.path.exists(outFile):
		print ("WARNING: File "+outFile+" does not exist.\nSkipping...");
		return [-1]*NUM_METRICS, [-1]*NUM_METRICS, -1
		#exit(-1)
	#else:
		#print ("Parsing outFile: " + outFile)
		
	n = -1
	gatherCnt=0
	
	with open(outFile) as f:
		line = f.readline()
		tokens = line.split()
		
		# for the first gather do not do anything
		if tokens[0]=="gather":
			gatherCnt += 1
		#
		
		if len(tokens)==0:	#found an empty line
			tokens = "0"
			#print(len(tokens))
		while tokens[0]!="gather" or gatherCnt!=2:
			line = f.readline();
			tokens = line.split()
			if len(tokens)==0:
				tokens = "0"
			if line=="":
				break
			if tokens[0]=="gather":
				gatherCnt += 1
			#print(tokens)
			#if tokens[0]=="numBlocks=":
			#	n = tokens[1]	
		
		metricNames = f.readline().split()
		line = f.readline()
		metricValues = [ float(x) for x in line.split()]
		
		
		# in case metrics are less than NUM_METRICS, fill the rest with -1
		for i in range(len(metricValues), NUM_METRICS):
			metricValues.append(-1)
			metricNames.append("-")
		print(metricNames)
		print(metricValues)
		
	return metricNames, metricValues, n
				
#######################################################################
# Special routine only for Geographers metric
# parses an outFile and returns the metric name found and the actual metric values

def parseOutFileForGeographer( outFile ):
	
	metricNames, metricValues, k = parseOutFile( outFile )
	
	numMetrics = len(metricNames)
	if numMetrics==NUM_METRICS:
		return metricNames, metricValues, k 
	
	retNames = []
	retValues = []
	
	for m in range(0, numMetrics):
		thisMetricName = metricNames[m] 
		thisMetricValues = metricValues[m]
		
		if thisMetricName in METRIC_NAMES:
			retNames.append( thisMetricName )
			retValues.append( thisMetricValues )
			
	return retNames, retValues, k
	
#######################################################################

def outFileString( exp, i, tool, outDir):
	if i>exp.size:
		print("WARNING: request for experiment run " + str(i) + " but experiment " +str(exp.ID) +" has size only " +str(exp.size) )
		return ""
	if tool not in allTools:
		print("WARNING: tool " + tool + " is invalid")
		return ""
	
	#outFile = os.path.basename(exp.graphs[i]).split('.')[0] + "Epsilon01_k" + str(exp.k[i]) +"_"+ tool + ".info"
	outFile = os.path.basename(exp.graphs[i]).split('.')[0] + "_k" + str(exp.k[i]) +"_"+ tool + ".info"
	#print(outFile)
	
	return os.path.join( outDir, tool , outFile)

#######################################################################

def printWarning( msg):
	frameinfo = getframeinfo(currentframe())
	print( frameinfo.filename +", " + str(frameinfo.lineno) + ": WARNING: " + msg)
		
def printError( msg):
	frameinfo = getframeinfo(currentframe())
	print( frameinfo.filename +", " + str(frameinfo.lineno) + ": ERROR" + msg)	