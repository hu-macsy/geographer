from subprocess import call
from submitFileWrapper_Supermuc import *
from header import*
from inspect import currentframe, getframeinfo
from meansHeader import *


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

def addRelativePlot( exp, metricValues, metricName, metricPos, toolNames, baseToolId, plotF):
	
	# m is the index of the wanted metric
	m = metricPos
	
	if baseToolId >= len(toolNames):
		print("ERROR: tool ID given " +  str(baseToolId) +" is too big, must be < " + str(len(toolNames)) )
		return -1;

	#plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=ratio , legend style={at={(1.5,0.7)}}, xmode = log, log basis x= 2, xtick={")
	plotF.write("\n\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=ratio, legend style={at={(1.5,0.7)}}, xmode = log, log basis x= 2, xtick={")
	for x in range(0, len(exp.k)-1 ):
		if exp.k[x]!=-1:
			plotF.write( str(exp.k[x]) +", ")
					
	if exp.k[-1]!=-1:
		plotF.write( str(exp.k[-1]) + "}]\n")
	else:
		plotF.write( "}]\n")
		
	#plotF.write("\\addplot plot coordinates{\n")

	for t in range(0,len(toolNames)):
		
		#if baseToolId==t:	# do not print the base tool metric
		#	continue
		
		thisToolMetrics = metricValues[t]
		thisToolName = toolNames[t]
		
		if baseToolId==t:
			plotF.write("\\addplot [mark=none] plot coordinates{\n")
		else:
			plotF.write("\\addplot plot coordinates{\n")
		
		
		for i in range(0,len(thisToolMetrics[m])):
			if thisToolMetrics[m][i]!=-1:
				# get the relative value, divide by the base tool metric
				if metricValues[baseToolId][m][i] == -1:
					plotF.write("("+str(exp.k[i])+", nan)\n")
				elif metricValues[baseToolId][m][i] == 0:	# avoid division with zero
					plotF.write("("+str(exp.k[i])+", "+ str(0) + ")\n")	
				else:
					plotF.write("("+str(exp.k[i])+", "+ str(thisToolMetrics[m][i]/metricValues[baseToolId][m][i]) + ")\n")	
				#print( metricValues[baseToolId][m][i] )
			else:
				plotF.write("("+str(exp.k[i])+", nan)\n")
		
		plotF.write("};\n")
		plotF.write("\\addlegendentry{"+thisToolName+"}\n")
		
	plotF.write("\\end{axis}\n\\end{tikzpicture}\n")


#---------------------------------------------------------------------------------------------	

def addGeoMeanInfo( metricValues, metricName, metricPos, toolNames, baseToolId, plotF):
	
	# one metric, for all tools
	geoMeanForAllTools = getGeomMeanForMetric( metricValues, metricName, metricPos, toolNames, baseToolId)
	
	if geoMeanForAllTools[0] == -1:
		print("Skipping the geometric mean for metric " + metricName);
		return -1;
	
	plotF.write("\nGeometric mean for metric: " + metricName + " , base tool: " + toolNames[baseToolId] +"\\\\ \n" )
	
	#print(geoMeanForAllTools)
	if geoMeanForAllTools==None:
		return [-1]*len(toolNames)
	
	for i in range(0, len(geoMeanForAllTools) ):
		plotF.write(toolNames[i] + " " + str(geoMeanForAllTools[i]) + "\\\\\n")
		
	return geoMeanForAllTools
	

#---------------------------------------------------------------------------------------------	

def addHarmMeanInfo( metricValues, metricName, metricPos, toolNames, baseToolId, plotF):
	
	# one metric, for all tools
	harmMeanForAllTools = getHarmMeanForMetric( metricValues, metricName, metricPos, toolNames, baseToolId)
	
	if harmMeanForAllTools[0] == -1:
		print("Skipping the harmonic mean for metric " + metricName);
		return -1;
	
	plotF.write("\nHarmonic mean for metric: " + metricName + " , base tool: " + toolNames[baseToolId] +"\\\\ \n" )
	
	for i in range(0, len(harmMeanForAllTools) ):
		plotF.write(toolNames[i] + " " + str(harmMeanForAllTools[i]) + "\\\\\n")
		
	return harmMeanForAllTools
		
#---------------------------------------------------------------------------------------------		
# len(metricValues) == len(tools), a numTools*numMetrics*exp.size 3D matrix
# len(metricValues[i]) == len(metricNames)
# metricValues is a list of lists of lists (3D array): 
#		metricValues[i] = a list of lists for all all metric values for tool i ,	size = num_of_tools
#		metricValues[i][j] = a list with values for tool i and metric j	,			size = num_of_metrics
#		metricValues[i][j][l] = the value for tool i, metric j, for exp.k[l] ,		size = size_of_experiment
		
def createPlotsForExp(exp, toolNames, metricNames, metricValues, gatherDir, outDir ):
	
	numMetrics = len(metricNames)
	#numMetrics =NUM_METRICS
	numTools = len(toolNames)
	print("number of metrics in createPlotsForExp is " + str(numMetrics)+" and number of tools " + str(numTools) )
	
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

	plotID = []
	for toolName in toolNames:
		for t in range(0, len(allTools)):
			if toolName== allTools[t]:
				plotID.append(t)
				
	for i in sorted(plotID):
		plotFileName += str(i)

	plotFile = os.path.join( outDir, plotFileName);
	
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
		#else, if option=="2", do not do anything
			
	plotFile += ".tex"
	#
	print(plotFile)
	#exit(-1)
	#
		
	with open(plotFile,'w') as plotF:
		plotF.write("\\documentclass{article}\n\\usepackage{tikz}\n\\usepackage{pgfplots}\n") 
		plotF.write("\\usepackage[total={7in, 9in}]{geometry}\n\n")
		plotF.write("\\begin{document}\n\n")
		plotF.write("\\today\n\n");
		
		plotF.write("Experiment type: ");
		if exp.expType==0:
			plotF.write(" weak\n\n")
		elif exp.expType==1:
			plotF.write(" strong\n\n")
		else:
			plotF.write(" other\n\n")
			
		plotF.write("Data gathered from directory: \n\n")
		plotF.write( gatherDir +"\n\n")
		plotF.write("From files: \n\n")
			
		for i in range(0,exp.size):
			#outF1 =  outFiles[i]
			outF1 = exp.graphs[i]
			outF2 = ""
			for c in outF1:
				if c!="_":
					outF2 += c
				else:
					outF2 = outF2+"\_"
						
			#plotF.write( outF2  +", ")
			plotF.write( outF2  +", k= " + str(exp.k[i]) +"\\\\")
			#if exp.expType==1 or exp.expType==2:
			#	break
					
		plotF.write("\\clearpage")
		
		#
		# plot each metric that is shared with the competitors in one figure
		#
		
		# these are 2D matrices with numRows= numMetrics and numColumns=numTools
		# geoMeanMatrix[i][j] will have the geomMean (of the ralative values compared to the base tool (toolNames[0]))
		# 		for tool i and metric j. The same for the harmonic mean
		geoMeanMatrix = []
		harmMeanMatrix = []
		metricNamesShort = []
		
		# for all metrics
		for m in range(0, numMetrics):
			metricName = metricNames[m]
			## TODO:
			## maybe skip some metrics for brevity??
			## for example imbalance
			
			#exclude metrics from the plot-for-all
			excludedMetrics=["imbalance", "maxBnd", "totBnd", 'maxBndPercnt', 'avgBndPercnt', '-']
						
			if metricName in excludedMetrics:
				print("Not calculating for metric " + metricName)
				continue
			
			print(">>> to plot for metric " + metricName)
			
			plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel= "+ METRIC_VALUES[m] +", legend style={at={(1.5,0.7)}},xmode = log, ymode=log, log basis x= 2, xtick={")
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
			
			addRelativePlot( exp, metricValues, metricName, m, toolNames, 0, plotF )
			plotF.write("\\caption{Metric: "+ metricName+"}\n\\end{figure}\n\n")
			
			# write the geom and harm mean as text
			addGeoMeanInfo( metricValues, metricName, m, toolNames, 0, plotF) 
			addHarmMeanInfo( metricValues, metricName, m, toolNames, 0, plotF)
						
			metricNamesShort.append( metricName )
			
			geoMeanMatrix.append( getGeomMeanForMetric( metricValues, metricName, m, toolNames, 0) )
			harmMeanMatrix.append( getHarmMeanForMetric( metricValues, metricName, m, toolNames, 0) )
						
			plotF.write("\\clearpage\n\n")
		
		plotF.write("\n\\begin{figure}\n")
		print( metricNamesShort )
		
		# plot for geometric mean
		plotBarMeanForAllTool( geoMeanMatrix, metricNamesShort, len(metricNamesShort), toolNames, plotF)
		plotF.write("\\caption{Geometric mean for all metrics and all tools for experiment:" + str(exp.ID) +" with base tool: " + wantedTools[0] +"}\n\\end{figure}\n\n")
		
		#plot for harmonic mean
		#plotF.write("\n\\begin{figure}\n")
		#plotBarMeanForAllTool( harmMeanMatrix, metricNamesShort, len(metricNamesShort), toolNames, plotF)
		#plotF.write("\\caption{Harmonic mean for all metrics and all tools for experiment:" + str(exp.ID) +" with base tool: " + wantedTools[0] +"}\n\\end{figure}\n\n")
		
		plotF.write("\n\n")
		
		#newNumMetrics = len(geoMeanMatrix)

		plotF.write("\n\n")
		plotF.write("\\end{document}")

	print("Plots written in file " + plotFile )

#---------------------------------------------------------------------------------------------		

def createMeanPlotsForAllExp( wantedExp, wantedTools, gatherDir, outDir, repartMetrics):
	
	numExp = len(wantedExp)
	
	# create the filename
	plotFileName = "expMeans_t"

	for t in wantedTools:
		for t2 in range(0,NUM_TOOLS):
			tool2 = allTools[t2]
			if t==tool2:
				plotFileName += str(t2)
				
	plotFileName += "_exp"
	for exp in wantedExp:
		plotFileName+=str(exp.ID)
	
	plotFileName += ".tex"
	plotFile = os.path.join( outDir, plotFileName);
		
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
		
		plotF.write("\\clearpage\n\n")

	for exp in wantedExp:	
		
		metricValues= []
		for tool in wantedTools:
			metricNames, metricValuesTmp = gatherExpTool( exp, tool, gatherDir, repartMetrics )
			metricValues.append( metricValuesTmp)
			
			
		numMetrics = len(metricNames)
		numTools = len(wantedTools)
		print("number of metrics in createMeanPlotsForAllExp is " + str(numMetrics)+" and number of tools " + str(numTools) )
		
		#print( str( len(metricValues) ) )
		#rint( str( len(metricValues[0]) ) )
		#print( str( len(metricValues[0][0]) ) )
		
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
		harmMeanMatrix = []
		metricNamesShort = []
		
		plotF= open(plotFile,'a')
		
		for m in range(0, numMetrics):
			metricName = metricNames[m]
		
			## skip certain metrics
			## or handle differently
			#if metricName=="imbalance":				
				#continue
			if metricName=="timeTotal":
				timeTmeans = getGeomMeanForMetric( metricValues, metricName, m, wantedTools, 0)
				plotF.write("Values for metric timeTotal: ")
				for t in range(0, len(timeTmeans)):
					plotF.write(wantedTools[t]+"= " + str(timeTmeans[t])+" , ")
					
			#exclude metrics from the plot-for-all
			excludedMetrics=["imbalance", "maxBnd", "totBnd", 'maxBndPercnt', 'avgBndPercnt', 'numDisBlocks']
						
			if metricName in excludedMetrics:
				print("Not calculating for metric " + metricName)
				continue
			
			metricNamesShort.append( metricName )
			
			geoMeanMatrix.append( getGeomMeanForMetric( metricValues, metricName, m, wantedTools, 0) )
			harmMeanMatrix.append( getHarmMeanForMetric( metricValues, metricName, m, wantedTools, 0) )
		
		
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
			#if exp.expType==1 or exp.expType==2:
			#	break
			
		plotF.write("\n\\begin{figure}\n")
		#plotMeanForAllTool( geoMeanMatrix, metricNames, numMetrics, wantedTools, plotF, "Geometric")
		plotBarMeanForAllTool( geoMeanMatrix, metricNamesShort, len(metricNamesShort), wantedTools, plotF)
		#plotMeanForAllTool( harmMeanMatrix, metricNamesShort, len(metricNamesShort), wantedTools, plotF, "Harmonic")
		plotF.write("\\caption{Geometric mean for all metrics and all tools for experiment:" + str(exp.ID) +" with base tool: " + wantedTools[0] +"}\n\\end{figure}\n\n")	
		plotF.write("\n\n\\clearpage\n\n")
		plotF.close()
		
		
	plotF= open(plotFile,'a')
	plotF.write("\n\n")
	plotF.write("\\end{document}")
	print("Collective results written in file " + plotFile)

#---------------------------------------------------------------------------------------------		
	
	
def createCsv(exp, toolNames, metricNames, metricValues):

	numMetrics = len(metricNames)
	#numMetrics =NUM_METRICS
	numTools = len(toolNames)
	print("number of metrics in createCsv is " + str(numMetrics)+" and number of tools " + str(numTools) )
	
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
	
	csvFileName = "exp"+str(exp.ID)+".csv"
	
#---------------------------------------------------------------------------------------------		
# gather info for an experiment for a tool

def gatherExpTool( exp, tool, gatherDir , repartMetrics ):
	
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
		
		gatherFile = outFileString( exp, i, tool, gatherDir)
		
		if not os.path.exists( os.path.join( gatherDir, tool) ):
			print("### ERROR: directory to gather information " + os.path.join( gatherDir, tool) + " does not exist.\nAborting..." )
			exit(-1)

		if not os.path.exists( gatherFile ):
			print("### WARNING: file to gather information " + gatherFile + " does not exist. Inserting dummy value -1.")
			metricValuesTmp = [-1 for x in range(0, NUM_METRICS)]
			#metricNames = ["-"]*NUM_METRICS
		else:
			if tool=="Geographer":
				metricNames, metricValuesTmp, k = parseOutFileForGeographer( gatherFile )
			else:
				#TODO: this is awful, fix properly (it happens because keyword "gather" appears two times
				#in the repartition files...)
				if repartMetrics:
					metricNames, metricValuesTmp, k = parseRepartFile( gatherFile )
				else:
					metricNames, metricValuesTmp, k = parseOutFile( gatherFile )
				
	
		#print( str(len(metricValuesTmp)) )
		
		allMetrics.append( metricValuesTmp )
		#print(metricNames)
		#print(metricValues)

	# convert to traspose so we have a list for every metric		
	numMetrics = len(metricNames)
	if len(allMetrics[0])!=numMetrics:
		print("### WARNING: num metrics mismatch in gatherExpTool for tool " + tool+ ", numMetrics = " + str(numMetrics) +", len(allMetrics[0])= " + str(len(allMetrics[0])) )
	if NUM_METRICS!=numMetrics:
		print("### WARNING: num metrics mismatch in gatherExpTool for tool " + tool+ ", numMetrics = " + str(numMetrics) + ", NUM_METRICS= " +str(NUM_METRICS) )
		
	metricValues = [None]*numMetrics
	for m in range(0,numMetrics):
		metricValues[m] = [row[m] for row in allMetrics]		
	
	return metricNames, metricValues
	


############################################################ 
# # # # # # # # # # # #  main

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Gather output information from file in specified folder. The folder must contail a gather.config that is created automatically by the submit script and list the output files for each experiment')
	parser.add_argument('--tools','-t' , type=str , nargs='*', default="Geographer", help='Name of the tools. It can be: ' + str(allTools) )
	parser.add_argument('--configFile','-c', default="SaGa.config", help='The configuration file. ')
	parser.add_argument('--wantedExp', '-we', type=int, nargs='*', metavar='exp', help='A subset of the experiments that will be submited.')
	parser.add_argument('--gatherDir', '-g', type=str, default=toolsPath, help='Optional folder to gather output. If none is given then the default is used as specified in the header/config file.\nDefault now is' + toolsPath)
	parser.add_argument('--outDir', '-o', type=str, help='Optional folder to store output. If none is given then the default is used as specified in the header/config file.\nDefault now is' + plotsPath)
	parser.add_argument('--repartMetrics', '-rm', type=bool, default=False, help="Flag to use metrics for repartition")
	
	args = parser.parse_args()

	wantedExpIDs = args.wantedExp
	configFile = args.configFile
	wantedTools = args.tools
	gatherDir = os.path.abspath( args.gatherDir)
	repartMetrics = args.repartMetrics
	
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

	if args.gatherDir:
		if not os.path.exists(gatherDir):
			print("Directory to gather info is missing.\nAborting...");
			exit(-1)

	outDir = "-"
	
	if args.outDir:
		outDir = args.outDir
	else:
		#outDir = plotsPath
		outDir = os.path.join( gatherDir, "plots" )
	
	if not os.path.exists(outDir):
		print("Creating folder " + outDir )
		os.makedirs( outDir)
		
	#
	# parse config file
	#

	allExperiments = parseConfigFile( configFile )

	numExps = len(allExperiments);

	# get the wanted experiments
	wantedExp = []

	if wantedExpIDs is None or len(wantedExpIDs)==0:
		wantedExp = allExperiments
	else:
		for i in range(0, numExps):
			if i in wantedExpIDs:
				wantedExp.append( allExperiments[i] )


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
			metricNames, metricValues = gatherExpTool( exp, tool, gatherDir, repartMetrics )
			
			# this means that gatherExpTool actually found the files to gather
			if not metricNames==[]:
				#TODO: this is a 3D matrix and starts to get ugly...
				allMetricsForAllTools.append( metricValues)
				foundTools.append( tool )
			
			#print(metricNames)
			#print(metricValues)
		
		createPlotsForExp(exp, foundTools, metricNames, allMetricsForAllTools,gatherDir, outDir)
		
	createMeanPlotsForAllExp( wantedExp, wantedTools, gatherDir, outDir, repartMetrics)
		
	exit(0)

