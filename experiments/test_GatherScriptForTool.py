from subprocess import call
from submitFileWrapper_Supermuc import *
from header import*
from inspect import currentframe, getframeinfo

import os
import math
import random
import re
import sys
import argparse
#import numpy as np



# addRelativePlot( exp, [[],[],[],..], "time", "tool1", plotF)
# exp: a struct experiment
# metricValues: a 2D matrix, len(metricValues)= exp.size, len(metricValues[i])

#addRelativePlot( exp, ... , "timeSpMV", plotF)

def addRelativePlot( exp, metricValues, metricName, toolNames, baseToolId, plotF):
	for m in range(0, NUM_METRICS):
		if metricName==METRIC_NAMES[m]:
			break;
	
	if m==NUM_METRICS:
		print("ERROR: metric " + metricName + " was not found")
		return -1
	# now m is the index of the wanted metric
	
	#baseMetric = []
	#wantedMetricForTools = []
	if baseToolId >= len(toolNames):
		print("ERROR: tool ID given " +  str(baseToolId) +" is too big, must be < " + str(len(toolNames)) )
		return -1;
	
	#for t in range(0,len(toolNames)):
		#thisMetric = metricValues[t][m]
		#if t>= len(allTools):
		#	 print("ERROR: wrong index " +str(t) +" for tool, len(allTools)= " + str(len(allTools)) )
		#if baseTool==toolNames[t]:
		#	 baseToolId = t
		#else:
		#	wantedMetricForTools.append(thisMetric)
		
	#if baseToolId==-1:
	#	print("ERROR: did not find base tool in given tools")
	#	return -1

	plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=ratio , legend style={at={(1.5,0.7)}}, xtick={")
	for x in range(0, len(exp.k)-1 ):
		if exp.k[x]!=-1:
			plotF.write( str(exp.k[x]) +", ")
					
	if exp.k[-1]!=-1:
		plotF.write( str(exp.k[-1]) + "}]\n")
	else:
		plotF.write( "}]\n")
		
	#plotF.write("\\addplot plot coordinates{\n")
	print( len(toolNames))
	for t in range(0,len(toolNames)):
		
		#if baseToolId==t:	# do not print the base tool metric
		#	continue
		
		thisToolMetrics = metricValues[t]
		thisToolName = toolNames[t]
		
		#frameinfo = getframeinfo(currentframe())
		#print(frameinfo.filename +", " + str(frameinfo.lineno) + " :: " + str(len(thisToolMetrics)) + " __ " + thisToolName )
		
		if baseToolId==t:
			plotF.write("\\addplot [mark=none] plot coordinates{\n")
		else:
			plotF.write("\\addplot plot coordinates{\n")
		
		
		for i in range(0,len(thisToolMetrics[m])):
			if thisToolMetrics[m][i]!=-1:
				# get the relative value, divide by the base tool metric
				plotF.write("("+str(exp.k[i])+", "+ str(thisToolMetrics[m][i]/metricValues[baseToolId][m][i]) + ")\n")	#
				#print( metricValues[baseToolId][m][i] )
			else:
				plotF.write("("+str(exp.k[i])+", nan)\n")
		
		plotF.write("};\n")
		plotF.write("\\addlegendentry{"+thisToolName+"}\n")
		
	plotF.write("\\end{axis}\n\\end{tikzpicture}\n")
	plotF.write("\\caption{Metric: "+ metricName+" relative to "+ toolNames[baseToolId] +"}\n\\end{figure}\n\n")		


#---------------------------------------------------------------------------------------------		
# len(metricValues) == len(tools)
# len(metricValues[i]) == len(metricNames)
# metricValues is a list of lists of lists (3D array): 
#		metricValues[i] = a list of lists for all all metric values for tool i ,	size = num_of_tools
#		metricValues[i][j] = a list with values for tool i and metric j	,			size = num_of_metrics
#		metricValues[i][j][l] = the value for tool i, metric j, for exp.k[l] ,		size = size_of_experiment
		
def createPlotsGeneric(exp, toolNames, metricNames, metricValues):
	
	numMetrics = len(metricNames)
	#numMetrics =NUM_METRICS
	numTools = len(toolNames)
	print("number of metrics in createPlotsGeneric is " + str(numMetrics)+" and number of tools " + str(numTools) )
	
	#
	print( str( len(metricValues) ) )
	print( str( len(metricValues[0]) ) )
	print( str( len(metricValues[0][0]) ) )
	#		
	
	if numMetrics!= len(metricValues[0]) :
		print("WARNING: len(metricNames)= "+ str(numMetrics) + " and len(metricValues) mismatch= " + str(len(metricValues[0])) )	
	if len(metricValues[0][0])!= exp.size :
		print("WARNING: len(metricNames)= "+ len(metricsValues[0][0]) + " and exp.size mismatch= " + str(exp.size) )
	if numTools!=len(metricValues):
		print("WARNING: len(metricValues)= "+ str(len(metricValues)) + " and numTools mismatch= " + str(numTools))	
	#
	# create time plots, figures in .tex file
	#
	
	plotFileName = "exp"+str(exp.ID)+"_"
	'''
	for t in range(0, len(allTools)):
		globTool = allTools[t]
		if toolNames[t]==globTool:
			plotFileName += str(t)
	'''
	plotID = []
	for toolName in toolNames:
		for t in range(0, len(allTools)):
			if toolName== allTools[t]:
				plotID.append(t)
				
	for i in sorted(plotID):
		plotFileName += str(i)

	plotFile = os.path.join( plotsPath, plotFileName);
	
	# check if file already exists
	if os.path.exists(plotFile + ".tex"):
		print("Plot file " + plotFile + " already exists.\n0: Abort \n1: Add random ending \n2: Overwrite")
		option = raw_input("Provide option: ")
		while not( option=="0" or option=="1" or option=="2"):
			option = raw_input("Provide option: ")
		
		if option=="0":
			return -1
		elif option=="1":
			r = random.randint(0,99)
			while os.path.exists( plotFile + "_"+str(r)+".tex"):
				r = random.randint(0,99)
				
			plotFile += "_"+str(r)
		#else, if option=="2", do not do anthing
			
	plotFile += ".tex"
	#
	print(plotFile)
	#exit(-1)
	#
	
	
	
	#plotFile = os.path.join( plotDir, "plots_new_"+str(exp.expType)+expNumber+".tex")
		
	with open(plotFile,'w') as plotF:
		plotF.write("\\documentclass{article}\n\\usepackage{tikz}\n\\usepackage{pgfplots}\n\\begin{document}\n\n")
			
		plotF.write("Experiment type: ");
		if exp.expType==0:
			plotF.write(" weak\n\n")
		elif exp.expType==1:
			plotF.write(" strong\n\n")
		else:
			plotF.write(" other\n\n")
			
		plotF.write("Data gathered from directory: \n\n")
		#plotF.write( os.path.join(os.getcwd(),gatherDir) +"\n\n")
		plotF.write( toolsPath +"\n\n")
		plotF.write("From files: \n")
			
		for i in range(0,exp.size):
			#outF1 =  outFiles[i]
			outF1 = exp.graphs[i]
			outF2 = ""
			for c in outF1:
				if c!="_":
					outF2 += c
				else:
					outF2 = outF2+"\_"
						
			plotF.write( outF2  +", ")
			if exp.expType==1 or exp.expType==2:
				break
			#print( outF2 )
				
		#
		# plot each metric that is shared with the competitors in one figure
		#
		
		plotF.write("\\clearpage")
		
		# for all metrics
		for m in range(0, numMetrics):
			metricName = metricNames[m]
			
			plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel= "+ METRIC_VALUES[m] +", legend style={at={(1.5,0.7)}}, xtick={")
			for x in range(0, len(exp.k)-1 ):
				if exp.k[x]!=-1:							### changed index from i to x
					plotF.write( str(exp.k[x]) +", ")
						
			if exp.k[-1]!=-1:
				plotF.write( str(exp.k[-1]) + "}]\n")
			else:
				plotF.write( "}]\n")
				
			#for all tools
			for t in range(0,numTools):
				thisToolName = toolNames[t]
				thisToolMetrics = metricValues[t]
				
				if len(thisToolMetrics)>numMetrics :
					print("WARNING: number of metrics mismatch: len(thisToolMetrics)= " + str(len(thisToolMetrics)) + ", numMetrics= " + str(numMetrics) )
					print("\tfor tool " + thisToolName + ", graph: " + exp.graphs[i] +", k= " + str(exp.k[i])+" and metric: " + metricName)
					print("\tSkipping this metric")

				plotF.write("\\addplot plot coordinates{\n")
				#for i in range(0,exp.size):
				#print("WARNING:" + str(len(thisToolMetrics[m])) )
				for i in range(0,len(thisToolMetrics[m])):
					if thisToolMetrics[m][i]!=-1:
						plotF.write("("+str(exp.k[i])+", "+ str(thisToolMetrics[m][i]) + ")\n")
					else:
						plotF.write("("+str(exp.k[i])+", nan)\n")
										
				plotF.write("};\n")
				plotF.write("\\addlegendentry{"+thisToolName+"}\n")
				
			plotF.write("\\end{axis}\n\\end{tikzpicture}\n")
			plotF.write("\\caption{Metric: "+ metricName+"}\n\\end{figure}\n\n")
			
			addRelativePlot( exp, metricValues, metricName, toolNames, 0, plotF)
			plotF.write("\\clearpage")
		
		#addRelativePlot( exp, metricValues, "timeSpMV", toolNames, 0, plotF)
		
		plotF.write("\\end{document}")

	print("Plots written in file " + plotFile )

#---------------------------------------------------------------------------------------------		
# gather info for an experiment for a tool

def gatherExpTool( exp, tool ):
	
	if not tool in allTools:
		print("Wrong tool name: " + str(tool) +". Choose from: ") 
		for x in allTools:
			print( "\t"+str(x) ) 
		print("Aborting...")
		exit(-1)
		
	allMetrics = []
	metricNames = []
	
	#print(exp.size, len(exp.k), len(exp.graphs))
	for i in range(0, exp.size):
		#
		#exp.printExp()
		#
		#gatherFileName = exp.graphs[i].split('.')[0] +"_"+tool+".info"
		#gatherPath = os.path.join( toolsPath, tool, gatherFileName)
		gatherFile = outFileString( exp, i, tool)
		
		if not os.path.exists( os.path.join( toolsPath, tool) ):
			print("### ERROR: directory to gather information " + os.path.join( toolsPath, tool) + " does not exist.\nAborting..." )
			exit(-1)

		if not os.path.exists( gatherFile ):
			print("### WARNING: file to gather information " + gatherFile + " does not exist. Inserting dummy value -1.")
			metricValuesTmp = [-1 for x in range(0, NUM_METRICS)]
			#metricNames = ["-"]*NUM_METRICS
		else:
			metricNames, metricValuesTmp, k = parseOutFile( gatherFile )
	
		#print( str(len(metricValuesTmp)) )
		
		allMetrics.append( metricValuesTmp )
		#print(metricNames)
		#print(metricValues)

	# convert to traspose so we have a list for every metric		
	numMetrics = len(metricNames)
	if len(allMetrics[0])!=numMetrics:
		print("WARNING: num metrics mismatch in gatherExpTool for tool " + tool+ ", numMetrics = " + str(numMetrics) +", len(allMetrics[0])= " + str(len(allMetrics[0])) )
	if NUM_METRICS!=numMetrics:
		print("WARNING: num metrics mismatch in gatherExpTool for tool " + tool+ ", numMetrics = " + str(numMetrics) + ", NUM_METRICS= " +str(NUM_METRICS) )
		
	metricValues = [None]*numMetrics
	for m in range(0,numMetrics):
		metricValues[m] = [row[m] for row in allMetrics]		
	
	return metricNames, metricValues
	


#---------------------------------------------------------------------------------------------		
# gather info for an experiment for a tool

def gatherCompetitor( exp, tool ):
	
	if not tool in allTools:
		print("Wrong tool name: " + str(tool) +". Choose from: ") 
		for x in allTools:
			print( "\t"+str(x) ) 
		print("Aborting...")
		exit(-1)
		
	allMetrics = []
	
	#print(exp.size, len(exp.k), len(exp.graphs))
	for i in range(0, exp.size):
		gatherFileName = exp.graphs[i].split('.')[0] +"_"+tool+".info"
		#gatherFileName = exp.graphs[i].split('.')[0]+".info"
		gatherPath = os.path.join( toolsPath, tool, gatherFileName)
		print( gatherFileName )
		
		if not os.path.exists( os.path.join( toolsPath, tool) ):
			print("### ERROR: directory to gather information " + os.path.join( toolsPath, tool) + " does not exist.\nAborting..." )
			exit(-1)
		if not os.path.exists(gatherPath):
			print("### WARNING: file to gather information " + gatherPath + " does not exist.")#\nAborting...")
			#metricNames = ["--"]*9
			metricValuesTmp = [-1 for x in range(0,9)]
			k=-1
			#continue
			#exit(-1)
		else:
			metricNames, metricValuesTmp, k = parseOutFile( gatherPath )
				
		#if not k==exp.k[i]:
			#print("### WARNING: number of blocks mismatch, parse.k= "+ str(k) +" exp.k= "+ str(exp.k[i]) )#\nAborting...")
			#print("the output file for Geographer or the competitor does not exist\n")
			#metricNames = ["--"]*9
			#metricValuesTmp = [0 for x in range(0,9)]			
			#k= max(k, exp.k[i])
			#exit(-1)
		
		allMetrics.append( metricValuesTmp )
		#print(metricNames)
		#print(metricValues)

	# convert to traspose so we have a list for every metric		
	numMetrics = len(metricNames)
	metricValues = [None]*numMetrics
	for m in range(0,numMetrics):
		metricValues[m] = [row[m] for row in allMetrics]		
	
	return metricNames, metricValues
			



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

#exit(0)
# debug
#for exp in wantedExp:
#	exp.printExp()
#

#
# gather info for each wanted experiment
#

for exp in wantedExp:
	allMetricsForAllTools = []
	
	# it can be that gather files for some tool are missing
	foundTools= [] 
	
	for tool in wantedTools:
		print ("Start gather experiments fot tool " + tool)		
		#metricName is a list of size numMetrics with all the metric names
		#metricValues is list of lists, or a 2D matrix:
		#	metricValues[i] = a list of size exp.size for metric i
		#	metricValues[i][j] = the value of metric i for experiment run j
		metricNames, metricValues = gatherExpTool( exp, tool )
		
		# this means that gatherExpTool actually found the files to gather
		if not metricNames==[]:
			#this is a 3D matrix and starts to get ugly...
			allMetricsForAllTools.append( metricValues)
			foundTools.append( tool )
		
		#print(metricNames)
		#print(metricValues)
	print(" " )
	createPlotsGeneric(exp, foundTools, metricNames, allMetricsForAllTools)
	
exit(0)

