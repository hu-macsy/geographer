from subprocess import call
from submitFileWrapper_Supermuc import *
from header import*

import os
import math
import random
import re
import sys
import argparse


# paths are in header.py

# all avaialble tools
#allTools = ["Geographer", "parMetisGeom", "parMetisGraph"]

# absolute paths for the executable of each tool
#geoExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/Diamerisi"
#parMetisExe = "/home/hpc/pr87si/di36qin/parco-repart/Implementation/parMetisExe"

# other paths
#basisPath = os.path.expanduser("~/parco-repart/Implementation/experiments/")
#competitorsPath = os.path.join( basisPath, "competitors" )




#---------------------------------------------------------------------------------------------
# choose with which tool to submit the experiments

def submitExp( tool, exp, runDir):

	if tool=="Geographer":
		submitGeographer(exp, runDir)
	elif tool=="parMetisGraph":
		geom = False
		submitParMetis(exp, geom)
	elif tool=="parMetisGeom":
		geom= True
		submitParMetis(exp, geom)
	else:
		print("First argument must be a tool name. Possible tool name are: Geographer, parMetisGraph, parMetisGeom.")
		print("They must be given with these exact names, you gave: " + str(tool) + "'n.Aborting...\n")
		return -1
		
#---------------------------------------------------------------------------------------------		
# submit an experiment with geographer. runDir is the directory from where to gather info

def submitGeographer(exp, runDir):
		
	gatherPath = os.path.join( runDir, 'gather.config' )	
	gatherFile = open( gatherPath,'w' )

	for path in exp.paths:
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
			
	gatherFile.write("experiment "+ str(e) +" type "+ str(exp.expType) + " size " + str(exp.size) + "\n")
	
	for i in range(0,exp.size):
		outFile = str(e)+str(i)+str(exp.expType)+"_"+ exp.graphs[i].split('.')[0] + "_k" + exp.k[i]+".info"
		outPath = os.path.join(runDir, outFile)

		# set parameters for every experiment
		params = defaultSettings()
		params += " --outFile="+ outPath
		params += " --dimensions="+ exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		#print(params)
			
		# add info in file gather.config
		gatherFile.write( outFile+ "\n" )
		
		#commandString = "../Diamerisi --graphFile="+ exp.paths[i]+ params
		commandString = geoExe + " --graphFile="+ exp.paths[i]+ params
		print( commandString )
		print( " " )
		
		submitFilename = "llsub-"+exp.graphs[i].split('.')[0]+"_k"+str(exp.k[i])+"_Geographer.cmd"
		submitfile = createLLSubmitFile( runDir, submitFilename, commandString, "00:10:00", int(exp.k[i]) )
		print( submitfile )
		#call(["llsubmit", submitfile])

#---------------------------------------------------------------------------------------------		


def submitParMetis(exp, geom):	
	
	for i in range(0,exp.size):
		
		params = ""
		if geom:
			outFile1 = exp.graphs[i].split('.')[0] + "_k" + exp.k[i]+"_parMetisGeom.info"
			outFile = os.path.join( competitorsPath, "parMetisGeom", outFile1)
			if not os.path.exists( os.path.join( competitorsPath, "parMetisGeom") ):
				print("WARNING: Output directory " + os.path.join( competitorsPath, "parMetisGeom") +" does not exist, job NOT submited.\n Aborting...")
				exit(-1)
			params += " --geom"
		else:
			outFile1 = exp.graphs[i].split('.')[0] + "_k" + exp.k[i]+"_parMetisGraph.info"
			outFile = os.path.join( competitorsPath, "parMetisGraph", outFile1)
			if not os.path.exists( os.path.join( competitorsPath, "parMetisGraph") ):
				print("WARNING: Output directory " + os.path.join( competitorsPath, "parMetisGraph") +" does not exist, job NOT submited.\n Aborting...")
				exit(-1)
			
		params += " --dimensions=" + exp.dimension
		params += " --fileFormat="+ exp.fileFormat
	
		commandString = parMetisExe + " --graphFile " + exp.paths[i] + params + " --outFile="+outFile
		if geom:
			submitFilename = "llsub-"+exp.graphs[i].split('.')[0]+"_k"+str(exp.k[i])+"_parMetisGeom.cmd"
		else:
			submitFilename = "llsub-"+exp.graphs[i].split('.')[0]+"_k"+str(exp.k[i])+"_parMetisGraph.cmd"
		submitfile = createLLSubmitFile( os.path.join( competitorsPath, "tmp") , submitFilename, commandString, "00:05:00", int(exp.k[i]) )
		call(["llsubmit", submitfile])
		
		
		
		
		
	
############################################################ 
# # # # # # # # # # # #  main

#print ('Argument List:' + str(sys.argv))

parser = argparse.ArgumentParser(description='Submit jobs in Supermuc batch job system  for the selected tool. The experiments must be stored in the given configuration file.')
parser.add_argument('--tool','-t' , default="Geographer", help='Name of the tool. It can be: Geographer, parMetisGraph, parMetisGeom.')
parser.add_argument('--configFile','-c', default="SaGa.config", help='The configuration file. ')
parser.add_argument('--wantedExp', '-we', type=int, nargs='*', metavar='exp', help='A subset of the experiments that will be submited.')
#parse.add_argument('--runDirName')

args = parser.parse_args()
print(args)

wantedExp = args.wantedExp
configFile = args.configFile
tool = args.tool
if not tool in allTools:
	print("Wrong tool name: " + str(tool) +". Choose from: ") 
	for x in allTools:
		print( "\t"+str(x) ) 
	print("Aborting...")
	exit(-1)
	
print(tool, configFile, wantedExp)

if not os.path.exists(configFile):
	print("Configuration file " + configFile + " does not exist. Aborting...")
	exit(-1)
else:
	print("Collecting information from configuration file: " + configFile )

#
# parse config file
#

inPath = ""
expTypeFlag = -1; # 0 for weak, 1 for strong
lineCnt = 0
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
			assert(int(exp.dimension)>0),"Wrong or missing dimension in line "+str(lineCnt)
			assert(int(exp.fileFormat)>=0),"Wrong or missing fileFormat in line "+str(lineCnt)
			expData = f.readline(); lineCnt +=1

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

			while expData[0]!="#":
				exp = experiment()
				exp.expType = 1
				exp.dimension = dimension
				exp.fileFormat = fileFormat
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


#
# recheck experiments and if all files exist
#

for exp in allExperiments:
	'''	
	for path in exp.paths:
		#graphPath =  os.path.join(inPath, graph)
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
	'''		
	for i in range(0, exp.size):
		if int(exp.k[i])/16 != int(int(exp.k[i])/16):
			print("WARNING: k= " + exp.k[i] + " is not a multiple of 16")
	if int(exp.dimension) <=0:
		print("WARNING: wrong value for dimension: " + str(exp.dimension))
	if int(exp.fileFormat) <0:
		print("WARNING: wrong value for fileFormat: " + str(exp.dimension))


#
# submit a job for every experiment
#

if wantedExp is None or len(wantedExp)==0:
#if len(wantedExp)==0:
	wantedExp = [x for x in range(0, len(allExperiments)) ]
	
print("About to submit the following experiments:")
totalSubmits = 0
for e in wantedExp:
	exp = allExperiments[e]
	print("\t\t### Experiment number: " + str(e) + " ###")
	exp.printExp()
	totalSubmits += exp.size
	for path in exp.paths:
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
print("in total "+ str(totalSubmits)+" submits")

#confirm = input("Submit experiments Y/N:")
confirm = raw_input("Submit experiments with >>> " + str(tool) +" <<< Y/N:")
while not(str(confirm)=="Y" or str(confirm)=="N" or str(confirm)=="y" or str(confirm)=="n"):
	#confirm= input("Please type Y or N ")		#python3
	confirm= raw_input("Please type Y/y or N/n: ")	
	
if str(confirm)=='N':
		#call( ["rm -rf", runDir] )
		print("Aborting...")
		exit(0)

# create the run directory and gather file
run = getRunNumber(basisPath)

runDir =  os.path.join(basisPath,"run"+str(run))
if not os.path.exists(runDir):
    os.makedirs(runDir)
else:
	print("WARNING: folder for run " + str(run) + " already exists. Danger of overwritting older data, aborting...")
	exit(-1)
	
for e in wantedExp:				#TODO: adapt so you can run a subset of all the experiments, e.g.: 1,3,7-10
	exp = allExperiments[e]
		
	submitExp(tool, exp, runDir)

exit(0)









