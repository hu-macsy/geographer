from subprocess import call
from submitFileWrapper_Supermuc import *
from header import*

import os
import math
import random
import re



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


dirString = os.path.expanduser("~/ParcoRepart/Implementation/meshes/")
basicPath = os.path.expanduser("~/ParcoRepart/Implementation/experiments/")
inPath = ""

##
configFile = os.path.join(basicPath,"SaGa.config")
##

if not os.path.exists(configFile):
	print("Configuration file " + configFile + " does not exist. Aborting...")
	exit(-1)


#
# parse config file
#

expTypeFlag = -1; # 0 for weak, 1 for strong
lineCnt = 0
experiments = []
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
			
		print(expHeader)
		
		# traverse the experiment header line to store parameters
		dimension = -1
		fileFormat = -1
		for i in range(1, len(expHeader) ):
			if expHeader[i]=="dim":
				dimension = expHeader[i+1]
			elif expHeader[i]=="fileFormat":
				fileFormat = expHeader[i+1]
			
		# create and store experiments in an array
		
		# for weak scaling
		if expHeader[0]=="weak":		
			exp = experiment()
			exp.expType = 0
			exp.dimension = dimension
			exp.fileFormat = fileFormat
			assert(exp.dimension>0),"Wrong or missing dimension in line "+str(lineCnt)
			assert(exp.fileFormat>0),"Wrong or missing fileFormat in line "+str(lineCnt)
			expData = f.readline(); lineCnt +=1

			while expData[0]!="#":
				expTokens = expData.split()
				for i in range(0, len(expTokens)-1):
					exp.graphs.append( expTokens[0] )
					exp.paths.append( os.path.join(inPath, expTokens[0]) )
					exp.k.append(expTokens[i+1])
					exp.size +=1
				expData = f.readline(); lineCnt +=1
				
			allExperiments.append(exp)
			
		# for strong scaling
		elif expHeader[0]=="strong":	
			expData = f.readline(); lineCnt +=1

			while expData[0]!="#":
				exp = experiment()
				exp.expType = 1
				exp.dimension = dimension
				exp.fileFormat = fileFormat
				assert(exp.dimension>0), "Wrong or missing dimension in line "+str(lineCnt)
				assert(exp.fileFormat>0),"Wrong or missing fileFormat in line "+str(lineCnt)
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
				assert(exp.dimension>0), "Wrong or missing dimension in line "+str(lineCnt)
				assert(exp.fileFormat>0),"Wrong or missing fileFormat in line "+str(lineCnt)
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

print("\nAll experiments:")
for exp in allExperiments:
	exp.printExp()


#
# recheck experiments and if all files exist
#

for exp in allExperiments:
	for path in exp.paths:
		#graphPath =  os.path.join(inPath, graph)
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
		
	if exp.dimension <=0:
		print("WARNING: wrong value for dimension: " + str(exp.dimension))
	if exp.fileFormat <=0:
		print("WARNING: wrong value for fileFormat: " + str(exp.dimension))


#
# submit a job for every experiment
#

# create the run directory and gather file
run = getRunNumber(basicPath)

runDir =  os.path.join(basicPath,"run"+str(run))
if not os.path.exists(runDir):
    os.makedirs(runDir)
else:
	print("WARNING: folder for run " + str(run) + " already exists. Danger of overwritting older data, aborting...")
	exit(0)
	
gatherPath = os.path.join( runDir, 'gather.config' )	
gatherFile = open( gatherPath,'w' )


#for exp in allExperiments:
for e in range(0,4):				#TODO: adapt so you can run a subset of all the experiments, e.g.: 1,3,7-10
	exp = allExperiments[e]
	gatherFile.write("experiment "+ str(e) +" type "+ str(exp.expType) + " size " + str(exp.size) + "\n")
	
	for i in range(0,exp.size):
		outFile = str(e)+str(i)+str(exp.expType)+"_"+ exp.graphs[i].split('.')[0] + "_k" + exp.k[i] + ".out"
		outPath = os.path.join(runDir, outFile)

		# set parameters for every experiment
		params = defaultSettings()
		params += " --outFile="+ outPath
		params += " --dimensions="+ exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		#print(params)
			
		# add info in file gather.config
		gatherFile.write( outFile+ "\n" )
		#gatherFile.write( "#\n")
		
		commandString = "Diamerisi --graphFile="+ exp.paths[i]+ params
		print( commandString )
		print( " " )
		
		submitFilename = "llsub-"+exp.graphs[i]+"_p"+str(exp.k[i])+".cmd"
		submitfile = createLLSubmitFile( runDir, submitFilename, commandString, "00:10:00", int(exp.k[i]) )
		#call(["llsubmit", submitfile])








