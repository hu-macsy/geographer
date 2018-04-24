from subprocess import call
from submitFileWrapper_Supermuc import *

import os
import math
import random

iterations = 7
dimension = 2

dirString = os.path.expanduser("/gpfs/work/pr87si/di36qin/meshes/")

graphNumber = 7
formatString = '%02d' % graphNumber
#fileString = "refinedtrace-"+'000'+formatString

fileString = "alyaTestCaseA"
#fileString = "hugebubbles-00020"
#fileString = "hugetric-00020"

fileFormat = 0	# 0:METIS, 6:Binary

#numPEs = [ 2048, 4096, 16384, 32768 ]
numPEs = [ 16, 32, 64, 128 ]

#----------------------------------------------

if fileFormat==0:
	fileEnding=".graph"
elif fileFormat==6:
	fileEnding=".bgf"
else:
	print("Unknown format "+ str(format)+ ", aborting.")
	exit(-1)

filename = os.path.join(dirString, fileString+fileEnding)


if not os.path.exists(filename):
	print(filename + " does not exist.")
	exit(-1)


for p in numPEs:
	
	randID = random.randint(0,99)
	outFile= os.path.join( os.path.expanduser("/gpfs/work/pr87si/di36qin/jobOutputs/"), "info_"+fileString+"_p"+str(p)+"_metis_"+str(randID) )

	if (p/16 != int(p/16)):
		print ("Number of PEs is not a multiple od 16, aborting...")
		exit(-1)

#        commandString = "./parMetisExe --graphFile " + str(filename) + " --dimensions 2" +" --fileFormat="+str(fileFormat)+ " --outFile="+outFile
#      	submitfile = createLLSubmitFile("llsub-"+str(fileString)+"_p"+str(p)+"_metis_"+str(randID)+".cmd", commandString, "00:40:00", p)
#        call(["llsubmit", submitfile])

	outFile= os.path.join( os.path.expanduser("/gpfs/work/pr87si/di36qin/jobOutputs/"), "info_"+fileString+"_p"+str(p)+"_metisGeom_"+str(randID) )
	commandString =  "./parMetisExe --graphFile " + str(filename) + " --dimensions 2" +" --fileFormat="+str(fileFormat)+ " --outFile="+outFile+" --geom"

	submitfile = createLLSubmitFile("llsub-"+str(fileString)+"_p"+str(p)+"_metisGeom_"+str(randID)+".cmd", commandString, "00:05:00", p)
	call(["llsubmit", submitfile])
            
