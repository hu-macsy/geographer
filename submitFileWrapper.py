SCAI_LIB="/home/kit/iti/cq6340/WAVE/scai_lama/install/lib/"
BOOST_LIB="/home/kit/iti/cq6340/boost_1_61_0/stage/lib"
JOB_DIR="/home/hpc/pr87si/di36sop/WAVE/ParcoRepart/Implementation"

def createMOABSubmitFile(filename, commandString, walltime, processors, memory):
    classString = "singlenode" if processors <= 16 else "multinode"
    with open(filename, 'w') as f:
        f.write("#!/bin/bash"+"\n")
        f.write("#MSUB -q "+classString+"\n")
        f.write("#MSUB -l nodes="+str(max(1,int(processors/16)))+":ppn=16"+"\n")
        f.write("#MSUB -l walltime="+walltime+"\n")
        f.write("#MSUB -l pmem="+memory+"\n")

        f.write("module load mpi/openmpi/2.0-gnu-5.2"+"\n")
        f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+SCAI_LIB+":"+BOOST_LIB+"\n")

        f.write("mpirun "+commandString+"\n")
    
    return filename

def createLLSubmitFile(filename, commandString, walltime, processors, memory):
    classstring = "general" if int(processors) > 512 else "test"
    
    with open(filename, 'w') as f:
        f.write("#! /usr/bin/ksh"+"\n")
        f.write("#@ shell = /usr/bin/ksh"+"\n")
        f.write("#@ job_type = parallel"+"\n")
        f.write("#@ initialdir="+JOB_DIR+"\n")
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
        f.write("cd "+JOB_DIR+"\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("mpiexec " + commandString+"\n")

    return filename

def assembleCommandString(partitioner, graphFile, processors, others=""):
    if partitioner.lower() == "parco":
        return "Katanomi" + " --graphFile "+graphFile+others
    elif partitioner.lower() == "metis":
        return "metisWrapper" + " " + str(graphFile)
    elif partitioner.lower() == "rcb":
        return "zoltanWrapper" + " " + graphFile + " rcb"
    elif partitioner.lower() == "multijagged":
        return "zoltanWrapper" + " " + graphFile+" multijagged"
    else:
        raise ValueError("Partitioner "+str(partitioner)+" not supported.")
