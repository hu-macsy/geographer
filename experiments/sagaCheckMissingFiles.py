from header import*

import os
import math
import random
import re
import sys
import argparse



############################################################ 
# # # # # # # # # # # #  main

#print ('Argument List:' + str(sys.argv))

parser = argparse.ArgumentParser(description='Gather output information from file in specified folder. The folder must contail a gather.config that is created automatically by the submit script and list the output files for each experiment')
parser.add_argument('--tools','-t' , type=str , nargs='*', default="Geographer", help='Name of the tools. It can be: Geographer, parMetisGraph, parMetisGeom.')
parser.add_argument('--xT', type=str, nargs='*', help="Name of tool to exclude, Works only with --tools=all.")
parser.add_argument('--configFile','-c', default="SaGa.config", help='The configuration file. ')
parser.add_argument('--wantedExp', '-we', type=int, nargs='*', metavar='exp', help='A subset of the experiments that will be submited.')

args = parser.parse_args()

wantedExpIDs = args.wantedExp
configFile = args.configFile
wantedTools = args.tools
excludeTools = args.xT

if wantedTools[0]=="all":
	wantedTools= []
	for tool in allTools:
		if tool not in excludeTools:
			wantedTools.append(tool)
	
for tool in wantedTools:
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

numExps = len(allExperiments);

# get the wanted experiments
wantedExp = []

#
print( wantedExpIDs )
#

if wantedExpIDs is None or len(wantedExpIDs)==0:
	wantedExp = allExperiments
else:
	for i in range(0, numExps):
		if i in wantedExpIDs:
			wantedExp.append( allExperiments[i] )

#exit(0)
# debug
#for exp in wantedExp:
#	exp.printExp()
#

#
# gather info for each wanted experiment
#

numMissingFiles = 0
for exp in wantedExp:
	
	for tool in wantedTools:
		#print ("Start checking files for tool " + tool)		
		
		for i in range(0, exp.size):	
			gatherFile = outFileString( exp, i, tool)
		
			if not os.path.exists( os.path.join( toolsPath, tool) ):
				print("### ERROR: directory to gather information " + os.path.join( toolsPath, tool) + " does not exist.\nAborting..." )
				exit(-1)

			if not os.path.exists( gatherFile ):
				print("### WARNING: file to gather information " + gatherFile + " does not exist.")
				numMissingFiles += 1
			#else:
			#	print("Found file " + gatherFile)
	
print("There are " + str(numMissingFiles) +" files missing");
exit(0)

