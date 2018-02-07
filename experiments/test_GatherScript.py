from subprocess import call
from submitFileWrapper_Supermuc import *
from header import*

import os
import math
import random
import re
import sys
import argparse
#import numpy as np




def addplot( exp, plotF, wantedMetric, competitorMetricName, metricNames, metricValues, competitorNames, metricNamesCompetitors, metricValuesCompetitors ):
	# addplot( "prelCut", ["all", "finalCut"] )
		
	ourMetricTmp = []
	competitorMetricTmp = []
	competitorNameTmp = []
			
	for m in range(0, numMetrics):
		if metricNames[m]==wantedMetric:
			ourMetricTmp = metricValues[m]
		#else:
		#	print("ERROR: Metric " + wantedMetric +" not found.\nAborting...")
		#	exit(-1)
		
	if len(ourMetricTmp)==0:
		print("ERROR: Metric " + wantedMetric +" not found.\nAborting...")
		exit(-1)
		
	numCompetitors = len(metricNamesCompetitors)
	
	for c in range(0, numCompetitors):
		competitorName = competitorNames[c]
		for j in range(0, len(metricNamesCompetitors[c])):
			#print( competitorMetricName+" __ " + metricNamesCompetitors[c][j] )
			if competitorMetricName==metricNamesCompetitors[c][j]:
				competitorMetricTmp.append( metricValuesCompetitors[c][j] )
				competitorNameTmp.append( competitorName )
				break
					
	#found at least one competitor with the same metric name
	if len(competitorMetricTmp)!=0:
		plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=value, legend style={at={(1.5,0.7)}}, xtick={")	
		for x in range(0, len(exp.k)-1 ):
			if( exp.k[x] != -1):
				plotF.write( str(exp.k[x]) +", ")
		if( exp.k[x] != -1):
			plotF.write(str(exp.k[-1]) + "}]\n")		
		else:
			plotF.write("}]\n")
			
		# plot our metric values
		plotF.write("\\addplot plot coordinates{\n")
		for i in range(0,exp.size):
			if( metricValues[m][i]==-1 ):
				plotF.write("("+str(exp.k[i]) +", nan)\n")
			else:
				plotF.write("("+str(exp.k[i]) +", "+ str(ourMetricTmp[i]) +")\n")
										
		plotF.write("};\n")
		plotF.write("\\addlegendentry{Geographer "+ str(wantedMetric) +"}\n")			
			
		if exp.size != len(competitorMetricTmp[c]):
			print("Size mismatch for competitor, exp.size= " +str(exp.size) + " and len(compMetric)= " + str(len(competitorMetricTmp[c])) );
			#exit(-1)
			
		#plot all competitors values
		for c in range(0, len(competitorMetricTmp) ):
			plotF.write("\\addplot plot coordinates{\n")
			for i in range(0,exp.size):
			#for i in range(0,len(competitorMetricTmp[c])):
				plotF.write("("+str(exp.k[i])+", "+ str(competitorMetricTmp[c][i]) + ")\n")
										
			plotF.write("};\n")
			plotF.write("\\addlegendentry{"+competitorNameTmp[c]+" "+ competitorMetricName+"}\n")
				
		plotF.write("\\end{axis}\n\\end{tikzpicture}\n")
		plotF.write("\\caption{Metric: Geographer "+ wantedMetric +"}\n\\end{figure}\n\n")




		
def createPlots(exp, metricNames, metricValues, competitorNames, metricNamesCompetitors, metricValuesCompetitors):
	#
	# create time plots, figures in .tex file
	#
	
	plotDir = os.path.join( gatherDir, "plots")
	if not os.path.exists( plotDir):
		os.makedirs( plotDir )
	else:
		print("WARNING: folder for plots " + plotDir + " already exists. Danger of overwritting older data.")
		cont = raw_input("Should I continue? y/n: ")
		while not( str(cont)=='y' or str(cont)=='Y' or str(cont)=='N' or str(cont)=='n' ):
			cont = raw_input("Please type Y/y or N/n: ")
			
		if cont=='N' or cont=='n':
			exit(-1)

	plotFile = os.path.join( plotDir, "plots_"+str(exp.expType)+expNumber+".tex")
		
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
		plotF.write( os.path.join(os.getcwd(),gatherDir) +"\n\n")
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
			print( outF2 )
				
			
		# plot running times just for Geographer
			
		plotF.write("\n\n\\begin{figure}[h]\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=seconds, legend style={at={(1.5,0.7)}}, xtick={")
		for x in range(0, len(exp.k)-1 ):
			print( exp.k[x] )
			if( exp.k[x] != -1):
				plotF.write( str(exp.k[x]) +", ")
		if( exp.k[-1] != -1):
			plotF.write(str(exp.k[-1]) + "}]\n")
		else:
			plotF.write("}]\n")
			
		for m in [0,1,2,3]:
			plotF.write("\\addplot plot coordinates{\n")
			for i in range(0, exp.size):
				#if( exp.k[i] != -1 ):
				#	plotF.write("("+str(exp.k[i]) +", "+ str(metricValues[m][i]) +")\n")
				#WARNING: this prevents printing zero values....
				if( metricValues[m][i]==-1 ):
					plotF.write("("+str(exp.k[i]) +", nan)\n")
				else:
					plotF.write("("+str(exp.k[i]) +", "+ str(metricValues[m][i]) +")\n")
			plotF.write("};\n")			
			plotF.write("\\addlegendentry{" + metricNames[m] +"}\n")
				
				
		plotF.write("\\end{axis}\n\\end{tikzpicture}\n")
		plotF.write("\\caption{ Running times for Geographer}\n\\end{figure}\n\n")
		
		# plot specific metric pairs
		addplot( exp, plotF, "prelCut", "finalCut", metricNames, metricValues, competitorNames, metricNamesCompetitors, metricValuesCompetitors )
		addplot( exp, plotF, "timeGeom", "timeTotal", metricNames, metricValues, competitorNames, metricNamesCompetitors, metricValuesCompetitors )
	
			
		#
		# plot each metric that is shared with the competitors in one figure
		#
		
		numMetrics = len(metricNames)
		numCompetitors = len(metricNamesCompetitors)
		
		for m in range(0, numMetrics):
			metricName = metricNames[m]
			ourMetricTmp = metricValues[m]
			competitorMetricTmp = []
			competitorNameTmp = []
			#print( metricName )
			#print( metricValues[m] )
						
			# for all competitors
			for c in range(0, numCompetitors):
				competitorName = competitorNames[c]
				# check all metrics for this competitor
				for j in range(0, len(metricNamesCompetitors[c])):
					if metricName==metricNamesCompetitors[c][j]:
						competitorMetricTmp.append( metricValuesCompetitors[c][j] )
						competitorNameTmp.append( competitorName )
						
			#found at least one competitor with the same metric name
			if len(competitorMetricTmp)!=0:
				plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=value, legend style={at={(1.5,0.7)}}, xtick={")
				for x in range(0, len(exp.k)-1 ):
					plotF.write( str(exp.k[x]) +", ")
				plotF.write( str(exp.k[-1]) + "}]\n")
					
				# plot our metric values
				plotF.write("\\addplot plot coordinates{\n")
				for i in range(0,exp.size):
					#plotF.write("("+str(exp.k[i])+", "+ str(ourMetricTmp[i]) + ")\n")
					if( metricValues[m][i]==-1 ):
						plotF.write("("+str(exp.k[i]) +", nan)\n")
					else:
						plotF.write("("+str(exp.k[i]) +", "+ str(ourMetricTmp[i]) +")\n")
										
				plotF.write("};\n")
				plotF.write("\\addlegendentry{Geographer}\n")
				
				#plot all competitors values
				for c in range(0, len(competitorMetricTmp) ):
					plotF.write("\\addplot plot coordinates{\n")
					#for i in range(0,exp.size):
					for i in range(0,len(competitorMetricTmp[c])):
						plotF.write("("+str(exp.k[i])+", "+ str(competitorMetricTmp[c][i]) + ")\n")
										
					plotF.write("};\n")
					plotF.write("\\addlegendentry{"+competitorNameTmp[c]+"}\n")
				
				plotF.write("\\end{axis}\n\\end{tikzpicture}\n")
				plotF.write("\\caption{Metric: "+ metricName+"}\n\\end{figure}\n\n")
		
		
		plotF.write("\\end{document}")

	print("Plots written in file " + plotFile )

#---------------------------------------------------------------------------------------------		
# len(metricValues) == len(tools)
# len(metricValues[i]) == len(metricNames)
# metricValues is a list of lists of lists (3D array): 
#		metricValues[i] = a list of lists for all all metric values for tool i
#		metricValues[i][j] = a list with values for tool i and metric j
#		metricValues[i][j][l] = the value for tool i, metric j, for exp.k[l]
		
def createPlotsGeneric(exp, toolNames, metricNames, metricValues):
	
	numMetrics = len(metricNames)
	numTools = len(toolNames)
	print("number of metrics in createPlotsGeneric is " + str(numMetrics)+" and number of tools " + str(numTools) )
		
	if numMetrics!= len(metricValues[0]) :
		print("WARNING: len(metricNames)= "+ str(numMetrics) + " and len(metricValues) mismatch= " + str(len(metricValues)) )	
	if numMetrics!= exp.size :
		print("WARNING: len(metricNames)= "+ str(numMetrics) + " and exp.size mismatch= " + str(len(exp.size)) )
	if numTools!=len(metricValues):
		print("WARNING: len(metricValues)= "+ str(numValues) + " and numTools mismatch= " + str(numTools))	
	#
	# create time plots, figures in .tex file
	#
	
	plotDir = os.path.join( gatherDir, "plots")
	if not os.path.exists( plotDir):
		os.makedirs( plotDir )
	else:
		print("WARNING: folder for plots " + plotDir + " already exists. Danger of overwritting older data.")
		cont = raw_input("Should I continue? y/n: ")
		while not( str(cont)=='y' or str(cont)=='Y' or str(cont)=='N' or str(cont)=='n' ):
			cont = raw_input("Please type Y/y or N/n: ")
			
		if cont=='N' or cont=='n':
			exit(-1)

	plotFile = os.path.join( plotDir, "plots_new_"+str(exp.expType)+expNumber+".tex")
		
	#
	exit(-1)
	#
		
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
		plotF.write( os.path.join(os.getcwd(),gatherDir) +"\n\n")
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
			print( outF2 )
				
		#
		# plot each metric that is shared with the competitors in one figure
		#
		
		
		for m in range(0, numMetrics):
			metricName = metricNames[m]
			
			#for all tools
			for t in range(0,numTools):
				thisToolName = toolNames[t]
				thisToolMetrics = metricValues[t]
				
				if len(thisToolMetrics)!=numMetrics:
					printError("WARNING: number of metrics mismatch")
				#print( metricName )
				#print( metricValues[m] )
						
				plotF.write("\n\n\\begin{figure}\n\\begin{tikzpicture}\n\\begin{axis}[xlabel=k, ylabel=value, legend style={at={(1.5,0.7)}}, xtick={")
				for x in range(0, len(exp.k)-1 ):
					if exp.k[i]!=-1:
						plotF.write( str(exp.k[x]) +", ")
						
				if exp.k[-1]!=-1:
					plotF.write( str(exp.k[-1]) + "}]\n")
				else:
					plotF.write( "}]\n")
				'''
				# plot our metric values
				plotF.write("\\addplot plot coordinates{\n")
				for i in range(0,exp.size):
					#plotF.write("("+str(exp.k[i])+", "+ str(ourMetricTmp[i]) + ")\n")
					if( metricValues[m][i]==-1 ):
						plotF.write("("+str(exp.k[i]) +", nan)\n")
					else:
						plotF.write("("+str(exp.k[i]) +", "+ str(ourMetricTmp[i]) +")\n")
										
				plotF.write("};\n")
				plotF.write("\\addlegendentry{Geographer}\n")
				'''
				

				plotF.write("\\addplot plot coordinates{\n")
				#for i in range(0,exp.size):
				printWarning( len(thisToolMetrics[m]) )
				for i in range(0,len(thisToolMetrics[m])):
					plotF.write("("+str(exp.k[i])+", "+ str(thisToolMetrics[i]) + ")\n")
										
				plotF.write("};\n")
				plotF.write("\\addlegendentry{"+thisToolName+"}\n")
				
			plotF.write("\\end{axis}\n\\end{tikzpicture}\n")
			plotF.write("\\caption{Metric: "+ metricName+"}\n\\end{figure}\n\n")
		
		
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
		exp.printExp()
		gatherFileName = exp.graphs[i].split('.')[0] +"_"+tool+".info"
		gatherPath = os.path.join( toolsPath, tool, gatherFileName)
		
		if not os.path.exists( os.path.join( toolsPath, tool) ):
			print("### ERROR: directory to gather information " + os.path.join( toolsPath, tool) + " does not exist.\nAborting..." )
			exit(-1)

		if not os.path.exists(gatherPath):
			print("### WARNING: file to gather information " + gatherPath + " does not exist. Inserting dummy value -1.")
			metricValuesTmp = [-1 for x in range(0, NUM_METRICS)]
		else:
			metricNames, metricValuesTmp, k = parseOutFile( gatherPath )

		
		allMetrics.append( metricValuesTmp )
		#print(metricNames)
		#print(metricValues)

	# convert to traspose so we have a list for every metric		
	numMetrics = len(metricNames)
	if NUM_METRICS!=numMetrics:
		print("WARNING: num metrics mismatch in gatherExpTool");
		
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
parser.add_argument('gatherDir', help='The path to the directory that has all the files from which the information will be gathered')

args = parser.parse_args()
gatherDir = args.gatherDir

if not os.path.exists( gatherDir ):
	print ("Path "+ gatherDir + " does not exist.\nAborting...")
	exit(-1)
else:
	if not os.path.isdir( gatherDir ):
		print ("Wrong directory " + gatherDir+ "\nAborting...")
		exit(-1)

gatherFile = os.path.join( gatherDir, "gather.config")
if not os.path.exists(gatherFile):
	print("Configuration file " + gatherFile + " does not exist in folder " + gatherDir + "\nAborting...")
	exit(-1)

#
# read info from the gather file and store all experiments
#
	
lineCnt=0
allExperiments = []
allOutFiles = []
		
with open(gatherFile,'r') as f:
		
	line = f.readline(); lineCnt+=1
	while line:
		
		print (">>>>> " + line)
		#if line=='':
		#	exit(0)
		tokens = line.split()
		
		if tokens[0]!="experiment":
			print ("Error in gather file, expecting to find keyword \'experiment\' in line " + str(lineCnt))
		
		# gather experiment information from header line
		exp = experiment()
		expNumber = tokens[1]
		
		for i in range(0,len(tokens)):
			if tokens[i]=="type":
				exp.expType = int(tokens[i+1])
				i+=1
			if tokens[i]=="size":
				exp.size = int(tokens[i+1])
				i+=1
		
		outFiles= [None]* exp.size
		exp.graphs = [None]* exp.size
		exp.k = [None]* exp.size
		
		for i in range(0, exp.size):
			newline = f.readline(); lineCnt+=1
			newtokens = newline.split()
			
			print(newtokens)
			outFiles[i] = newtokens[0];
			#outFiles[i] = f.readline().strip('\n'); 
			#TODO: here, graphs have a different usage than in submit script, maybe change that
			exp.k[i] = newtokens[1]
			print(outFiles[i])
			print( exp.k[i])
			#exit(-2)
			exp.graphs[i] = outFiles[i][4:]
			#print (str(i) + ", line: " + str(lineCnt) + " __ " + exp.graphs[i].strip('\n') )
			
		allOutFiles.append( outFiles )
		allExperiments.append( exp )
			
		line = f.readline(); lineCnt+=1
	#while line
#with open(file)
	
	
#
# gather info for each experiment
#

for e in range(0, len(allExperiments)):
	print("Start gathering info for experiment " + str(e))
	exp = allExperiments[e]
	#exp.k = [None]* exp.size
	metricNames = []
	metricPerRun = []
	
	for i in range(0, exp.size):
		# parse out file to get metrics
		metricNamesTmp, metricValuesTmp, nada = parseOutFile( os.path.join( os.getcwd(), gatherDir, allOutFiles[e][i]) )

		#print( exp.k[i])
		#if i==0:
		if metricNames==[]:
			metricNames = metricNamesTmp
		else:
			for j in range(0, len(metricNames)):
				if metricNames[j]!=metricNamesTmp[j]:
					print("Wrong metric name " + metricNamesTmp[j]+ ".\nAborting...")
					exit(-1)
		
		if metricValuesTmp==[]:
			metricPerRun.append([-1 for x in range(0,13)])
		else:
			metricPerRun.append( metricValuesTmp )
		
	numMetrics = len(metricNames)
	#print(numMetrics)
	
	# convert to traspose so we have a list for every metric
		
	metricValues = [None]*numMetrics
	for m in range(0,numMetrics):
		metricValues[m] = [row[m] for row in metricPerRun]
	'''	
	print ("k: " + str([x for x in exp.k]) )
	for m in range(0,numMetrics):
		print ("Metric "+str(m)+": "+ metricNames[m])
		print ([ "%.2f"% x for x in metricValues[m] ])
	'''
		
	#
	# get info for competitors
	#
		
	#
	metricNamesPMGra, metricValuesPMGra = gatherCompetitor( exp, "parMetisGraph" )
	#
	metricNamesPMGeo, metricValuesPMGeo = gatherCompetitor( exp, "parMetisGeom" )
	#
	metricNamesPMSfc, metricValuesPMSfc = gatherCompetitor( exp, "parMetisSfc" )
	#
	metricNamesGeoSfc, metricValuesGeoSfc = gatherCompetitor( exp, "geoSfc")
	
	print( str(len(metricValuesPMGeo[0])) + " ++ " + str(len(metricValues[0])) )
	print(metricValuesPMGeo[0])
	print( str(exp.size) + " __ \n")
	print(metricValues[0] )
	print(metricNames)
	
	competitorNames = ["parMetisGraph", "parMetisGeom", "geomSfc"]
	#competitorNames = allTools[1:]
	metricNamesCompetitors = [ metricNamesPMGra, metricNamesPMGeo, metricNamesPMSfc, metricNamesGeoSfc]
	metricValuesCompetitors = [ metricValuesPMGra, metricValuesPMGeo, metricValuesPMSfc, metricValuesGeoSfc]
		
	# assure all tools/competitors have same length of matrics and same metric names
	for i in range(0, len(metricNamesCompetitors)):
		for j in range(i, len(metricNamesCompetitors)):
			if len(metricNamesCompetitors[i]) != len(metricNamesCompetitors[i]):
				print("WARNING: metric length mismatch");
			for mN in range(0, metricNamesCompetitors[i]):
				if metricNamesCompetitors[i][mN]!=metricNamesCompetitors[j][mN]:
					print("WARNING: metric name mismatch") 
		
	metricNumPM = len(metricNamesPMGra)
	if not metricNumPM==len(metricNamesPMGeo):
		print("WARNING: number of metrics for parMetisGraph and parMetisGeom must be the same")
		
	for i in range(0, metricNumPM):
		name = metricNamesPMGra[i]
		if name != metricNamesPMGeo[i]:
			print("WARNING: metric name " + name + " is different for parMetisGraph and parMetisGeom")
			
	
	'''
	for i in range(0, numMetrics):
		metricName = metricNames[i]
		print(metricName)
		print( metricValues[i] )
		for j in range(0, len(metricValuesPMGeo)):
			if metricName==metricNamesPMGeo[j]:
				print("parMetisGeom")
				print(metricValuesPMGeo[j])
				print("parMetisGraph")
				print(metricValuesPMGra[j])
	'''
				
	# create time plots, figures in .tex file in path runX/plots		
	createPlots(exp, metricNames, metricValues, competitorNames, metricNamesCompetitors, metricValuesCompetitors)
			
	#toolNames = competitorNames		
	#createPlotsGeneric(exp, toolNames, metricNames, metricValuesCompetitors)
			
	#addplot( "prelCut", ["all", "finalCut"] )
	#addplot( "timeGeom", ["all", "timeTotal"] )
		
exit(0)




	
	
