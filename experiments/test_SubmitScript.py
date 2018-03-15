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

	
def submitExp( exp, tool, outDir ):

	if tool=="Geographer":
		submitGeographer(exp, "graph", outDir)
	elif tool=="geoKmeans":
		submitGeographer(exp, "geoKmeans", outDir)
	elif tool=="geoSfc":
		submitGeographer(exp, "geoSfc", outDir)
	else:
		submitCompetitor( exp, tool, outDir)


#---------------------------------------------------------------------------------------------		

def submitGeographer(exp, version, outDir):
		
	for path in exp.paths:
		if not os.path.exists( path ) :
			print("WARNING: " + path + " does not exist.")
			
	params = "-1"
	tool = "-1"
	
	for i in range(0,exp.size):
		
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

		if not os.path.exists( os.path.join( outDir, tool) ):
			print("WARNING: Output directory " + os.path.join( outDir, tool) +" does not exist. Creating directory " + os.path.join( outDir, tool))
			os.makedirs( os.path.join( outDir, tool) )
			
		outFile = outFileString( exp, i, tool, outDir)
		
		if outFile=="":
			print( "outFile is empty for tool " + tool + " and experiment " + str(exp.ID) + "\n. Skippong job ...")
			return -1
		if os.path.exists( outFile ):
			print("\t\tWARNING: The outFile: " + outFile + " already exists, job NOT submitted.")
			continue
		
		params += " --epsilon=" + str(epsilon)
		params += " --dimensions="+ exp.dimension
		params += " --fileFormat="+ exp.fileFormat
				
		if exp.coordFormat!=-1:
			params += " --coordFormat="+ str(exp.coordFormat)
		
		if  exp.coordPaths is not None and len( exp.coordPaths)>0:
			if exp.coordPaths[i]!="-":
				params += " --coordFile=" + exp.coordPaths[i]
		
		params += " --outFile="+ outFile
		
		
		if not os.path.exists( executable):
			print("Executable " + executable + " does not exist.\nSkiping job submission")
			return -1
		
		commandString = executable + " --graphFile="+ exp.paths[i]+ params
		
		print(commandString)
		
		submitFilename = "llsub-"+os.path.basename(exp.graphs[i]).split('.')[0]+"_k"+str(exp.k[i])+"_"+version+".cmd"
		submitfile = createLLSubmitFile( os.path.join( outDir, "tmp"), submitFilename, commandString, "00:20:00", int(exp.k[i]) )
		#print( submitfile )
		call(["llsubmit", submitfile])
		
#---------------------------------------------------------------------------------------------		

def submitAllCompetitors( exp, outDir):
	for i in range(0,exp.size):
		
		graphName = os.path.basename(exp.graphs[i]).split('.')[0]
		submitFlag = 0
		
		for tool in allCompetitors:
			if not os.path.exists( os.path.join( outDir, tool) ):
				print("WARNING: Output directory " + os.path.join( outDir, tool) +" does not exist. Creating directory " + os.path.join( outDir, tool))
				os.makedirs( os.path.join( outDir, tool) )
				#print("WARNING: Output directory " + os.path.join( outDir, tool) +" does not exist, experiment NOT submited.\n Aborting...")
				#exit(-1)
			#outFile = os.path.join( outDir, tool, graphName)
			outFile = outFileString( exp, i, tool, outDir)
		
			if os.path.exists( outFile ):
				submitFlag += 1
		'''
		if submitFlag == NUM_COMPETITORS:
			print("\t\tWARNING: The graph: " + exp.graphs[i] + " for k=" + str(exp.k[i]) +" has an outFile for all tools, job NOT submitted.")
			continue
		'''	
		params = " --epsilon=" + str(epsilon)			
		params += " --dimensions=" + exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		params += " --outPath=" + outDir +"/"
		params += " --graphName=" + graphName 
		
		if exp.coordFormat!=-1:
			params += " --coordFormat="+ str(exp.coordFormat)
		
		if  exp.coordPaths is not None and len( exp.coordPaths)>0:
			if exp.coordPaths[i]!="-":
				params += " --coordFile=" + exp.coordPaths[i]
		
		if not os.path.exists( allCompetitorsExe):
			print("Executable " + allCompetitorsExe + " does not exist.\nSkiping job submission")
			return -1

		commandString = allCompetitorsExe + " --graphFile " + exp.paths[i] + params
		print(commandString)
		submitFilename = "llsub-"+ os.path.basename(exp.graphs[i]).split('.')[0]+"_k"+str(exp.k[i])+"_allCompetitors.cmd"
		submitfile = createLLSubmitFile( os.path.join( outDir, "tmp") , submitFilename, commandString, "00:20:00", int(exp.k[i]) )
		call(["llsubmit", submitfile])

#---------------------------------------------------------------------------------------------		

def submitAllRepart( exp, outDir ):
	for i in range(0,exp.size):
		
		graphName = os.path.basename(exp.graphs[i]).split('.')[0]
		submitFlag = 0
		
		for tool in allCompetitors:
			if not os.path.exists( os.path.join( outDir, tool) ):
				print("WARNING: Output directory " + os.path.join( outDir, tool) +" does not exist, experiment NOT submited.\n Aborting...")
				exit(-1)
			#outFile = os.path.join( outDir, tool, graphName)
			outFile = outFileString( exp, i, tool, outDir)
		
			if os.path.exists( outFile ):
				submitFlag += 1
		'''
		if submitFlag == NUM_COMPETITORS:
			print("\t\tWARNING: The graph: " + exp.graphs[i] + " for k=" + str(exp.k[i]) +" has an outFile for all tools, job NOT submitted.")
			continue
		'''	
		params = " --epsilon=" + str(epsilon)			
		params += " --dimensions=" + exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		params += " --outPath=" + outDir +"/"
		params += " --graphName=" + graphName 
		
		if exp.coordFormat!=-1:
			params += " --coordFormat="+ str(exp.coordFormat)
		
		if  exp.coordPaths is not None and len( exp.coordPaths)>0:
			if exp.coordPaths[i]!="-":
				params += " --coordFile=" + exp.coordPaths[i]
		
		if not os.path.exists( repartAllExe):
			print("Executable " + repartAllExe + " does not exist.\nSkiping job submission")
			return -1

		commandString = repartAllExe + " --graphFile " + exp.paths[i] + params
		print(commandString)
		submitFilename = "llsub-"+ os.path.basename(exp.graphs[i]).split('.')[0]+"_k"+str(exp.k[i])+"_repartAll.cmd"
		submitfile = createLLSubmitFile( os.path.join( outDir, "tmp") , submitFilename, commandString, "00:20:00", int(exp.k[i]) )
		call(["llsubmit", submitfile])

#---------------------------------------------------------------------------------------------	
def submitCompetitor(exp, tool, outDir):	
	
	for i in range(0,exp.size):
		
		params = ""
							
		outFile = outFileString( exp, i, tool, outDir)
		
		if outFile=="":
			print( "outFile is empty for tool " + tool + " and experiment " + str(exp.ID) + "\n. Skippong job ...")
			return -1
		
		if not os.path.exists( os.path.join( outDir, tool) ):
			print("WARNING: Output directory " + os.path.join( outDir, tool) +" does not exist, experiment NOT submited.\n Aborting...")
			exit(-1)
		
		if os.path.exists( outFile ):
			print("\t\tWARNING: The outFile: " + outFile + " already exists, job NOT submitted.")
			continue
		
		params += " --epsilon=" + str(epsilon)
		params += " --dimensions=" + exp.dimension
		params += " --fileFormat="+ exp.fileFormat
		
		if exp.coordFormat!=-1:
			params += " --coordFormat="+ str(exp.coordFormat)
		
		if  exp.coordPaths is not None and len( exp.coordPaths)>0:
			if exp.coordPaths[i]!="-":
				params += " --coordFile=" + exp.coordPaths[i]
				
		if not os.path.exists( competitorsExe):
				print("Executable " + competitorsExe + " does not exist.\nSkiping job submission")
				return -1
	
		commandString = competitorsExe + " --tool " + tool + " --graphFile " + exp.paths[i] + params + " --outFile="+outFile

		submitFilename = "llsub-"+ os.path.basename(exp.graphs[i]).split('.')[0]+"_k"+str(exp.k[i])+"_" + tool+".cmd"
		submitfile = createLLSubmitFile( os.path.join( outDir, "tmp") , submitFilename, commandString, "00:10:00", int(exp.k[i]) )
		#call(["llsubmit", submitfile])
		



	
	
############################################################ 
# # # # # # # # # # # #  main

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Submit jobs in Supermuc batch job system  for the selected tools. The experiments must be stored in the given configuration file.')
	parser.add_argument('--tools','-t' , type=str , nargs='*', default="Geographer", help='Name of the tools. It can be: Geographer, parMetisGraph, parMetisGeom.')
	parser.add_argument('--configFile','-c', default="SaGa.config", help='The configuration file. ')
	parser.add_argument('--wantedExp', '-we', type=int, nargs='*', metavar='exp', help='A subset of the experiments that will be submited.')
	parser.add_argument('--outDir', '-o', type=str, help='Optional folder to store output. If none is given then the default is used as specified in the header/config file.')

	args = parser.parse_args()
	#print(args)

	wantedExp = args.wantedExp
	configFile = args.configFile
	wantedTools = args.tools
		
	for tool in wantedTools:
		if tool=="all":
			continue
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
			
	outDir = "-"
	
	if args.outDir:
		outDir = args.outDir
		if not os.path.exists(outDir):
			os.makedirs(outDir)
		# this is needed to store the submit file
		if not os.path.exists( os.path.join(outDir, "tmp") ):
			os.makedirs(  os.path.join(outDir, "tmp")  )
	else:
		outDir = toolsPath

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
		
		
	if wantedTools[0]=="all":
		#wantedTools = allTools[1:]
		print("\n\tWARNING: Will call allCompetitorsExe that runs the experiment with all tools!!")
		confirm = raw_input("Continue? :")
		while not(str(confirm)=="Y" or str(confirm)=="N" or str(confirm)=="y" or str(confirm)=="n"):
			#confirm= input("Please type Y or N ")		#python3
			confirm= raw_input("Please type Y/y or N/n: ")	
			
		if str(confirm)=='N' or str(confirm)=='n':
			print("Not submitting experiments, aborting...")
			exit(0)
			
		for e in wantedExp:	
			exp = allExperiments[e]
			submitAllCompetitors( exp, outDir )	
			#submitAllRepart( exp, outDir)
			
		exit(0)
		
	for tool in wantedTools:	
		
		confirm = raw_input("Submit experiments with >>> " + str(tool) +" <<< Y/N:")
		while not(str(confirm)=="Y" or str(confirm)=="N" or str(confirm)=="y" or str(confirm)=="n"):
			#confirm= input("Please type Y or N ")		#python3
			confirm= raw_input("Please type Y/y or N/n: ")	
			
		if str(confirm)=='N' or str(confirm)=='n':
				print("Not submitting experiments...")
				continue
				#exit(0)
			
		for e in wantedExp:	
			exp = allExperiments[e]

			submitExp( exp, tool, outDir )

	print("Exiting submit script")
	exit(0)









