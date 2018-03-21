import random
import os

SCAI_LIB="/home/kit/iti/op4308/scai_lama/install/lib/"
BOOST_LIB="/home/kit/iti/op4308/Code/boost_1_63_0/stage/lib"

#MKL_LIB="MKL_LIB_COM /opt/bwhpc/common/compiler/intel/compxe.2016.4.258/lib/intel64"


def createMOABSubmitFile(filename, commandString, walltime, processors, memory, mapFlag):
	
	if processors<=28:
		if processors<=16:
			classString= "singlenode"
		else:
			classString= "fat"
	else:
		classString= "multinode"

	jobName = str(filename[5:-4])

	with open(filename, 'w') as f:
		f.write("#!/bin/bash"+"\n")
		f.write("#MSUB -q "+classString+"\n")
		f.write("#MSUB -l walltime="+walltime+"\n")
		f.write("#MSUB -l pmem="+memory+"\n")
		f.write("#MSUB -o job_"+jobName+".output"+"\n")
	
	flag = 0
	for i in range(28, 0, -1):
		if processors%i==0:
			flag = 1
			f.write("#MSUB -l nodes="+ str(processors/i) +":ppn="+ str(i) +"\n")
			break

	if flag==0:
		raise ValueError("Number of processors "+ str(processors) +" could not be divided into less than 28 processes per node.")

	f.write("#MSUB -m ea\n")
	f.write("#MSUB -M charilaos.tzovas@kit.edu\n")

	f.write("module load mpi/openmpi/2.0-gnu-5.2"+"\n")
	f.write("module load numlib/mkl/11.3.4"+"\n")
	f.write("module load devel/ddt/6.1.2"+"\n")
	f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+SCAI_LIB+":"+BOOST_LIB+"\n")

	#disabling map since the licences are limited and jobs tend to fail
	if processors<=256 and mapFlag:
		# -o the name of output .map file.
		f.write("map -profile -o "+ jobName + " mpirun "+commandString+"\n")
	else:
		f.write(" mpirun "+commandString+"\n")
	
	return filename


# for supermuc

JOB_DIR="/home/hpc/pr87si/di36qin/parco-repart/Implementation/experiments"
SM_SCAI_LIB="/home/hpc/pr87si/di36qin/Lama/lib"
SM_BOOST_LIB="/home/hpc/pr87si/di36qin/Code/boost_1_63_0/stage/lib"
METIS_LIB="/home/hpc/pr87si/di36qin/parco-repart/metis/lib"
PARMETIS_LIB="/home/hpc/pr87si/di36qin/parco-repart/parmetis/lib"
ARGTABLE_LIB="/home/hpc/pr87si/di36qin/argtable/build/lib/"		#needed for ParHip

def createLLSubmitFile( path, filename, commandString, walltime, processors):

	if int(processors) <=512:
		classstring = "test"
	elif int(processors) > 8192:
		classstring = "large"
	else:
		classstring = "general"
	#classstring = "fat"

	jobName = str(filename[6:-4])
	
	nodes= 1
	flag = 0
	for i in range(16, 0, -1):
		if processors%i==0:
			flag = 1
			nodes = processors/i;
			#f.write("#MSUB -l nodes="+ str(processors/i) +":ppn="+ str(i) +"\n")
			break

	if flag==0:
		raise ValueError("Number of processors "+ str(processors) +" could not be divided into less than 28 processes per node.")

	fullPath = os.path.join( path, filename)
	parPath = os.path.abspath( os.path.dirname(path) )

	with open(fullPath, 'w') as f:
		f.write("#! /bin/bash"+"\n")
		#f.write("#@ shell = /usr/bin/bash"+"\n")
		f.write("#@ job_type = parallel"+"\n")
		f.write("#@ initialdir="+parPath+"/jobOutputs \n")
		f.write("#@ job_name ="+jobName+"\n")
		f.write("#@ class = "+classstring+"\n")
		f.write("#@ node_usage = not_shared"+"\n")
		f.write("#@ wall_clock_limit = "+walltime+"\n")
		f.write("#@ network.MPI = sn_all,,us,,"+"\n")
		f.write("#@ output = "+jobName+".out\n")
		f.write("#@ error = "+jobName+".err\n")
		f.write("#@ node = "+ str(int(nodes))+"\n")
	
		if nodes<128 :
			f.write("#@ island_count=1"+"\n")
		elif nodes<=512:
			f.write("#@ island_count=1,2"+"\n")
		elif nodes<=1024:
			f.write("#@ island_count=2,3"+"\n")
		elif nodes<=1536:
			f.write("#@ island_count=3,4"+"\n")
		else:
			f.write("#@ island_count=4,5"+"\n")

		f.write("#@ total_tasks = "+str(processors)+"\n")
		f.write("#@ notification=always\n")
		f.write("#@ notify_user=ctzovas@uni-koeln.de\n")      

		#	not sure if it should be used
		#	energyTag = jobName.replace("-","_")	#not allowed character
		energyTag = "NONE"
		f.write("#@ energy_policy_tag="+energyTag+"\n")
		f.write("#@ minimize_time_to_solution=yes\n")

		f.write("#@ queue"+"\n")        
	
		f.write(". /etc/profile\n")
		f.write(". /etc/profile.d/modules.sh\n")
		f.write("cd "+JOB_DIR+"\n")
		f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+SM_SCAI_LIB+":"+SM_BOOST_LIB+":"+METIS_LIB+":"+PARMETIS_LIB+":"+ARGTABLE_LIB+"\n")
		f.write("export MP_SINGLE_THREAD=no \n")
		f.write("export OMP_NUM_THREADS=1\n")
		f.write("export MP_TASK_AFFINITY=core:$OMP_NUM_THREADS\n")
		f.write("mpiexec -n "+ str(processors) + " " + commandString+"\n")

	return fullPath

def assembleCommandString(partitioner, graphFile, others=""):
	if partitioner.lower() == "parco":
		return "Katanomi" + " --graphFile="+graphFile + others
	elif partitioner.lower() == "metis":
		return "parMetisExe --graphFile=" + graphFile + others
	elif partitioner.lower() == "testinitial":
		return "testInitial" + " --graphFile=" + str(graphFile) + others
	elif partitioner.lower() == "diamerisi":
		return "Diamerisi --graphFile=" + str(graphFile) + others
	else:
		raise ValueError("Partitioner "+str(partitioner)+" not supported.")
