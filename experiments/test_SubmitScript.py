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

def submitExp( tool, exp):

	if tool=="Geographer":
		submitGeographer_noGather(exp, "graph")
	elif tool=="geoKmeans":
		submitGeographer_noGather(exp, "geoKmeans")
	elif tool=="geoSfc":
		submitGeographer_noGather(exp, "geoSfc")
	elif tool=="parMetisGraph":
		geom = 0
		submitParMetis(exp, geom)
	elif tool=="parMetisGeom":
		geom= 1
		submitParMetis(exp, geom)
	elif tool=="parMetisSfc":
		geom= 2
		submitParMetis(exp, geom)
	else:
		print("First argument must be a tool name. Possible tool name are: Geographer, parMetisGraph, parMetisGeom.")
		print("They must be given with these exact names, you gave: " + str(tool) + "'n.Aborting...\n")
		return -1
		
#---------------------------------------------------------------------------------------------		
# submit an experiment with geographer. runDir is the directory from where to gather info

def submitGeographer(exp, version):
		
	'''
	runDir =  os.path.join(basisPath,"run"+str(run))
	if not os.path.exists(runDir):
		os.makedirs(runDir)
	else:
		print("WARNING: folder for run " + str(run) + " already exists. Danger of overwritting older data, aborting...")
		exit(-1)
	'''
	# create a gather.config file
	toolDir= os.path.join( toolsPath, version )
	gatherPath = os.path.join( toolDir, 'gather_'+ str(exp.ID)+'.config' )	
	gatherFile = open( gatherPath,'a' )
	
	for path in exp.paths:
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
			
	gatherFile.write("experiment "+ str(e) +" type "+ str(exp.expType) + " size " + str(exp.size) + "\n")
	
	for i in range(0,exp.size):
		#outFile = str(e)+str(i)+str(exp.expType)+"_"+ exp.graphs[i].split('.')[0] +"_k" + exp.k[i]+ "_geo"+version.capitalize() + ".info"
		outFile = exp.graphs[i].split('.')[0] +"_k" + exp.k[i]+ "_" + version + ".info"
		outPath = os.path.join(toolDir, outFile)
		
		if os.path.exists( outPath):
			print("\t\tWARNING: The outFile: " + outPath + " already exists, job NOT submitted.")
			continue
		
		# add info in file gather.config
		gatherFile.write( outFile+ " " + str(exp.k[i]) + "\n" )
		
		if version=="graph":
			# set parameters for every experiment
			params = defaultSettings()
			executable = geoExe
		elif version=="geoKmeans":
			executable = initialExe
			params = " --initialPartition=3"
		elif version=="geoSfc":
			executable = initialExe
			params = " --initialPartition=0"
		else:
			print("Version given: " + version + " is not implemented.\nAborting...");
			exit(-1)
		
		params += " --outFile="+ outPath
		params += " --dimensions="+ exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		#print(params)
		
		commandString = executable + " --graphFile="+ exp.paths[i]+ params
		
		submitFilename = "llsub-"+exp.graphs[i].split('.')[0]+"_k"+str(exp.k[i])+"_"+version+".cmd"
		submitfile = createLLSubmitFile( os.path.join( toolsPath, "tmp"), submitFilename, commandString, "00:10:00", int(exp.k[i]) )
		#print( submitfile )
		call(["llsubmit", submitfile])

#---------------------------------------------------------------------------------------------		

def submitGeographer_noGather(exp, version):
		
	for path in exp.paths:
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
			
	params = "-1"
	tool = "-1"
	
	if version=="graph":
		# set parameters for every experiment
		params = defaultSettings()
		executable = geoExe
		tool = "Geographer"
	elif version=="geoKmeans":
		executable = initialExe
		params = " --initialPartition=3"
		tool = version
	elif version=="geoSfc":
		executable = initialExe
		params = " --initialPartition=0"
		tool = version
	else:
		print("Version given: " + version + " is not implemented.\nAborting...");
		exit(-1)			
	
	for i in range(0,exp.size):
		outFile = outFileString( exp, i, tool)
		
		if outFile=="":
			print( "outFile is empty for tool " + tool + " and experiment " + str(exp.ID) + "\n. Skippong job ...")
			return -1
		if os.path.exists( outFile ):
			print("\t\tWARNING: The outFile: " + outFile + " already exists, job NOT submitted.")
			continue
		
		params += " --outFile="+ outFile
		params += " --dimensions="+ exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		#print(params)

		commandString = executable + " --graphFile="+ exp.paths[i]+ params
		
		submitFilename = "llsub-"+exp.graphs[i].split('.')[0]+"_k"+str(exp.k[i])+"_"+version+".cmd"
		submitfile = createLLSubmitFile( os.path.join( toolsPath, "tmp"), submitFilename, commandString, "00:10:00", int(exp.k[i]) )
		#print( submitfile )
		call(["llsubmit", submitfile])

#---------------------------------------------------------------------------------------------		


def submitParMetis(exp, geom):	
	
	for i in range(0,exp.size):
		
		params = ""
		parMetisVersion = ""
		
		if geom==0:
			tool = "parMetisGraph"
		elif geom==1:
			tool = "parMetisGeom"
		elif geom==2:
			tool = "parMetisSfc"
		else:
			print("Wrong value geom= " +str(geom) +"\nAborting...")
			exit(-1)
			
		outFile = outFileString( exp, i, tool)
		
		if outFile=="":
			print( "outFile is empty for tool " + tool + " and experiment " + str(exp.ID) + "\n. Skippong job ...")
			return -1
		
		if not os.path.exists( os.path.join( toolsPath, parMetisVersion) ):
			print("WARNING: Output directory " + os.path.join( toolsPath, "parMetisGraph") +" does not exist, experiment NOT submited.\n Aborting...")
			exit(-1)
		
		if os.path.exists( outFile ):
			print("\t\tWARNING: The outFile: " + outFile + " already exists, job NOT submitted.")
			continue
		
		params += " --geom " +str(geom)						
		params += " --dimensions=" + exp.dimension
		params += " --fileFormat="+ exp.fileFormat
	
		commandString = parMetisExe + " --graphFile " + exp.paths[i] + params + " --outFile="+outFile

		submitFilename = "llsub-"+exp.graphs[i].split('.')[0]+"_k"+str(exp.k[i])+"_" + parMetisVersion+".cmd"
		submitfile = createLLSubmitFile( os.path.join( toolsPath, "tmp") , submitFilename, commandString, "00:05:00", int(exp.k[i]) )
		call(["llsubmit", submitfile])
		

	
	
############################################################ 
# # # # # # # # # # # #  main

#print ('Argument List:' + str(sys.argv))

parser = argparse.ArgumentParser(description='Submit jobs in Supermuc batch job system  for the selected tools. The experiments must be stored in the given configuration file.')
parser.add_argument('--tools','-t' , type=str , nargs='*', default="Geographer", help='Name of the tools. It can be: Geographer, parMetisGraph, parMetisGeom.')
parser.add_argument('--configFile','-c', default="SaGa.config", help='The configuration file. ')
parser.add_argument('--wantedExp', '-we', type=int, nargs='*', metavar='exp', help='A subset of the experiments that will be submited.')
#parse.add_argument('--runDirName')

args = parser.parse_args()
print(args)

wantedExp = args.wantedExp
configFile = args.configFile
tools = args.tools

if tools[0]=="all":
	tools = allTools
	
for tool in tools:
	if not tool in allTools:
		print("Wrong tool name: " + str(tool) +". Choose from: ") 
		for x in allTools:
			print( "\t"+str(x) ) 
		print("Aborting...")
		exit(-1)
	

if not os.path.exists(configFile):
	print("Configuration file " + configFile + " does not exist. Aborting...")
	exit(-1)
else:
	print("Collecting information from configuration file: " + configFile )

#
# parse config file
#
allExperiments = parseConfigFile( configFile )

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
	wantedExp = [x for x in range(0, len(allExperiments)) ]
	
for tool in tools:	
	
	# print output
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


	confirm = raw_input("Submit experiments with >>> " + str(tool) +" <<< Y/N:")
	while not(str(confirm)=="Y" or str(confirm)=="N" or str(confirm)=="y" or str(confirm)=="n"):
		#confirm= input("Please type Y or N ")		#python3
		confirm= raw_input("Please type Y/y or N/n: ")	
		
	if str(confirm)=='N' or str(confirm)=='n':
			#call( ["rm -rf", runDir] )
			print("Aborting...")
			continue
			#exit(0)

	# create the run directory and gather file
	#run = getRunNumber(basisPath)
		
	for e in wantedExp:	
		exp = allExperiments[e]
			
		submitExp(tool, exp)

print("Exiting submit script")
exit(0)









