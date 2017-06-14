
# coding: utf-8

def createMOABSubmitFile(filename, commandString, walltime, processors, memory):   
    with open(filename, 'w') as f:
        f.write("#!/bin/bash"+"\n")
        f.write("#MSUB -q multinode"+"\n")
        f.write("#MSUB -l nodes="+str(max(1,int(processors/16)))+":ppn=16"+"\n")
        f.write("#MSUB -l walltime="+walltime+"\n")
        f.write("#MSUB -l pmem="+memory+"\n")

        f.write("module load mpi/openmpi/2.0-gnu-5.2"+"\n")
        f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kit/iti/cq6340/WAVE/scai_lama/install/lib/:/home/kit/iti/cq6340/boost_1_61_0/stage/lib\n")

        f.write("mpirun "+commandString+"\n")
    
    return filename


# In[ ]:

def createLLSubmitFile(filename, commandString, walltime, processors, memory):
    classstring = "general" if int(processors) > 512 else "test"
    
    with open(filename, 'w') as f:
        f.write("#! /usr/bin/ksh"+"\n")
        f.write("#@ shell = /usr/bin/ksh"+"\n")
        f.write("#@ job_type = parallel"+"\n")
        f.write("#@ initialdir=/home/hpc/pr87si/di36sop/WAVE/ParcoRepart/Implementation"+"\n")
        f.write("#@ job_name = LLRUN"+"\n")
        f.write("#@ class = general"+"\n")
        f.write("#@ node_usage = not_shared"+"\n")
        f.write("#@ wall_clock_limit = "+walltime+"\n")
        f.write("#@ network.MPI = sn_all,,us,,"+"\n")
        f.write("#@ notification = never"+"\n")
        f.write("#@ output = LLRUN.out.$(jobid)"+"\n")
        f.write("#@ error =  LLRUN.err.$(jobid)"+"\n")
        f.write("#@ node = 1"+"\n")
        f.write("#@ island_count=1,1"+"\n")
        f.write("#@ total_tasks = "+str(processors)+"\n")
        f.write("#@ queue"+"\n")
        
        f.write(". /etc/profile\n")
        f.write(". /etc/profile.d/modules.sh\n")
        f.write("cd /home/hpc/pr87si/di36sop/WAVE/ParcoRepart/Implementation\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("mpiexec " + commandString+"\n")

    return filename


# In[ ]:

def assembleCommandString(partitioner, graphFile, processors, others=""):
    if partitioner.lower() == "parco":
        return "Katanomi" + " --graphFile "+graphFile+" --dimensions 2"+others
    elif partitioner.lower() == "metis":
        return "metisWrapper" + " " + str(graphFile)
    #else if partitioner.lower() == "rcb":
    #    return graphFile+" rcb"
    #else if partitioner.lower() == "multijagged":
    #    return graphFile+" multijagged"
    else:
        raise ValueError("Partitioner "+str(partitioner)+" not supported.")


# In[ ]:

from subprocess import call
import os
import math
import random

iterations = 5
dimension = 2
minExp = -1.5
maxExp = 1

dirString = os.path.expanduser("~/WAVE/Giesse-Repart/mesh-sequences")

graphNumber = 1
formatString = '%02d' % graphNumber
p = 16*2**graphNumber
filename = os.path.join(dirString, "refinedtrace-"+'000'+formatString+".graph")
n = 4500000*2**graphNumber

scalingFactor = math.pow(float(n / p), float(1)/dimension)

for i in range(iterations):
        minBorderNodes = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
        stopAfterNoGainRounds = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
        minGainForNextGlobalRound = int(10**(random.uniform(minExp, maxExp))*scalingFactor)
        gainOverBalance = random.randint(0,1)
        skipNoGainColors = random.randint(0,1)
        tieBreakingStrategy = random.randint(0,2)
        useDiffusionTieBreaking = int(tieBreakingStrategy == 1)
        useGeometricTieBreaking = int(tieBreakingStrategy == 2)
        multiLevelRounds = random.randint(0,7)

        if not os.path.exists(filename):
                print(filename + " does not exist.")
        else:
            others = " --minBorderNodes="+str(minBorderNodes)
            others += " --stopAfterNoGainRounds="+str(stopAfterNoGainRounds)
            others += " --minGainForNextGlobalRound="+str(minGainForNextGlobalRound)
            others += " --gainOverBalance="+str(gainOverBalance)
            others += " --skipNoGainColors="+str(skipNoGainColors)
            others += " --useDiffusionTieBreaking="+str(useDiffusionTieBreaking)
            others += " --useGeometricTieBreaking="+str(useGeometricTieBreaking)
            others += " --multiLevelRounds="+str(multiLevelRounds)
            commandString = assembleCommandString("parco", filename, p, others)
            submitfile = createMOABSubmitFile("msub-"+str(p)+"-"+str(i)+".cmd", commandString, "00:10:00", p, "4000mb")
            call(["msub", submitfile])

                

