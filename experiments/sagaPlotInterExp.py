from submitFileWrapper_Supermuc import *
from header import*
from inspect import currentframe, getframeinfo
from meansHeader import *
from test_GatherScriptForTool import gatherExpTool

import os
import math
import random
import re
import sys
import argparse

def createInterExpPlots( wantedExp, wantedTools):
	
	numExp = len(wantedExp)
	numTools = len(wantedTools)
	
	print("numExp= " + str(numExp) + " and numTools= " + str(numTools) )
		
	# a 3D matrix: 	geoMeanForAllExps[i]: a 2D matrix for experiment i
	#				geoMeanForAllExps[i][j]: a list for experiment i and tool j
	#				geoMeanForAllExps[i][j][m]: the value for experiment i, tool j, metric m
	#geoMeanForAllExps = []
	
	#geoMeanForAllExps = [[] for i in range(0,numTools]
	
	for e in range(0, numExp):
		exp = wantedExp[e]	
		
		metricValues= []
		for tool in wantedTools:
			metricNames, metricValuesTmp = gatherExpTool( exp, tool )
			metricValues.append( metricValuesTmp)
		
		numMetrics = len(metricNames)
		numTools = len(wantedTools)
		
		if numMetrics!= len(metricValues[0]) :
			print("WARNING: len(metricNames)= "+ str(numMetrics) + " and len(metricValues) mismatch= " + str(len(metricValues[0])) )	
		if len(metricValues[0][0])!= exp.size :
			print("WARNING: len(metricNames)= "+ len(metricsValues[0][0]) + " and exp.size mismatch= " + str(exp.size) )
		if numTools!=len(metricValues):
			print("WARNING: len(metricValues)= "+ str(len(metricValues)) + " and numTools mismatch= " + str(numTools))	
			
		# these are 2D matrices with numRows= numMetrics and numColumns=numTools
		# geoMeanMatrix[i][j] will have the geomMean (of the relative values compared to the base tool (wantedTools[0]))
		# 		for tool i and metric j. The same for the harmonic mean
		geoMeanMatrix = []
		validMetrics = 0
		
		for m in range(0, numMetrics):
			metricName = metricNames[m]
		
			## skip certain metrics
			## or handle differently
			if metricName=="imbalance":				
				continue
			'''
			if metricName=="timeTotal":
				timeTmeans = getGeomMeanForMetric( metricValues, metricName, wantedTools, 0)
				plotF.write("Values for metric timeTotal: ")
				for t in range(0, len(timeTmeans)):
					plotF.write(wantedTools[t]+"= " + str(timeTmeans[t])+" , ")
			'''
			geoMeanMatrix.append( getGeomMeanForMetric( metricValues, metricName, wantedTools, 0) )
			validMetrics += 1
			
		# initialize if it is the first experiment or multiply else
		if e==0:
			geoMeanForAllExps = geoMeanMatrix
		else:
			for i in range(0,validMetrics):
				for j in range(0, numTools):
					geoMeanForAllExps[i][j] *= geoMeanMatrix[i][j]
		
	#for e in wantedExp
	
	# get the root of each entry
	for i in range(0,validMetrics):
		for j in range(0,numTools ):
			geoMeanForAllExps[i][j] = geoMeanForAllExps[i][j]**(1/float(numExp) )
	
		
	# create the filename
	plotFileName = "expMeansInterExp_t"
	#plotFileName += "t"
	for t in wantedTools:
		for t2 in range(0,NUM_TOOLS):
			tool2 = allTools[t2]
			if t==tool2:
				plotFileName += str(t2)
				
	plotFileName += "_exp"
	for exp in wantedExp:
		plotFileName+=str(exp.ID)
	
	plotFileName += ".tex"
	plotFile = os.path.join( plotsPath, plotFileName);
	
	if os.path.exists(plotFile + ".tex"):
		print("WARNING: Plot file " + plotFile + " already exists and will be overwriten.")
		
	# write file header
	with open(plotFile,'w') as plotF:
		plotF.write("\\documentclass{article}\n\\usepackage{tikz}\n\\usepackage{pgfplots}\n") 
		plotF.write("\\usepackage[total={7in, 9in}]{geometry}\n\n")
		plotF.write("\\begin{document}\n\n")
		plotF.write("\\today\n\n");
	
		plotF.write("Data gathered from directory: \n\n")
		plotF.write( toolsPath +"\n\n")
		#plotF.write("From files: \n")
		#TODO: add all the files?
			
		for exp in wantedExp:
			plotF.write("\n\nPlots for experiment with id:" + str(exp.ID) + " containing instances:\\\\")
			for i in range(0,exp.size):
				outF1 = exp.graphs[i]
				outF2 = ""
				for c in outF1:
					if c!="_":
						outF2 += c
					else:
						outF2 = outF2+"\_"
							
				plotF.write( outF2  +", k= " + str(exp.k[i]) +"\\\\")
				
		plotF.write("\n\\begin{figure}\n")
		plotMeanForAllTool( geoMeanForAllExps, metricNames, numMetrics, wantedTools, plotF, "Geometric")
		plotF.write("\\caption{Geometric mean for all metrics and all tools for different experiments:" + str(exp.ID) +" with base tool: " + wantedTools[0] +"}\n\\end{figure}\n\n")	
		#plotF.write("\n\n\\clearpage\n\n")
		plotF.write("\\end{document}")
		
	print("Collective results written in file " + plotFile)
		
############################################################ 
# # # # # # # # # # # #  main

#print ('Argument List:' + str(sys.argv))

parser = argparse.ArgumentParser(description='Gather output information from file in specified folder. The folder must contail a gather.config that is created automatically by the submit script and list the output files for each experiment')
parser.add_argument('--tools','-t' , type=str , nargs='*', default="Geographer", help='Name of the tools. It can be: Geographer, parMetisGraph, parMetisGeom.')
parser.add_argument('--configFile','-c', default="SaGa.config", help='The configuration file. ')
parser.add_argument('--wantedExp', '-we', type=int, nargs='*', metavar='exp', help='A subset of the experiments that will be submited.')

args = parser.parse_args()

wantedExpIDs = args.wantedExp
configFile = args.configFile
wantedTools = args.tools

if wantedTools[0]=="all":
	wantedTools = allTools[1:]		# skipping Geographer
	
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

createInterExpPlots( wantedExp, wantedTools)