#!/usr/bin/python3

import argparse
import subprocess
import sys
import time


subprocess.call("make -C build -j 12 install", shell=True)

# "installation/..." is the installation path

parser = argparse.ArgumentParser()
parser.add_argument('--executable', '-x', type=str, help="Path to the geographer executable, usually in install_path/bin/GeographerStandalone" )
parser.add_argument('--verbose', '-v', dest='verbose', default=False, action='store_true', help="Show output of executions")

args = parser.parse_args()

ex = args.executable
verb = args.verbose

if ex is None:
    ex = "installation/bin/GeographerStandalone"

allCommands = [ 
"mpirun -n 4 "+ ex + " --graphFile meshes/rotation-00000.graph --noRefinement",
"mpirun -n 7 "+ ex + " --graphFile meshes/slowrot-00000.graph --noRefinement",
"mpirun -n 4 "+ ex + " --graphFile meshes/slowrot-00000.graph --noRefinement --blockSizesFile testing/blockSizes_k4_w1.txt",
"mpirun -n 7 "+ ex + " --graphFile meshes/rotation-00000.graph --localRefAlgo parMetisRefine",
# the autoSetCpuMem option does not work
#"mpirun -n 9 "+ ex + " --graphFile meshes/rotation-00000.graph --noRefinement --autoSetCpuMem --processPerNode 3",
"mpirun -n 11 "+ ex + " --graphFile meshes/rotation-00000.graph --initialPartition geoSFC --localRefAlgo parMetisRefine",
"mpirun -n 6 "+ ex + " --graphFile meshes/rotation-00000.graph --initialPartition geoSFC --localRefAlgo geographer --metricsDetail all",
"mpirun -n 6 "+ ex + " --graphFile meshes/rotation-00000.graph --initialPartition geoKMeansBalance --metricsDetail easy",
"mpirun -n 6 "+ ex + " --graphFile meshes/rotation-00000-random-3.graph --initialPartition geoKMeansBalance --metricsDetail no --noRefinement",
"mpirun -n 6 "+ ex + " --graphFile meshes/rotation-00000-random-3.graph --initialPartition geoKMeansBalance --metricsDetail easy --numNodeWeights 2 --noRefinement",
"mpirun -n 6 "+ ex + " --graphFile meshes/rotation-00000-random-3.graph --initialPartition geoKMeans --metricsDetail easy --numNodeWeights 2 --noRefinement",
"mpirun -n 12 "+ ex + " --graphFile meshes/rotation-00000-random-3.graph --initialPartition geoHierRepart --hierLevels 3,2,2 --metricsDetail easy --numNodeWeights 2 --noRefinement",
"mpirun -n 9 "+ ex + " --graphFile meshes/rotation-00000-random-3.graph --initialPartition geoHierRepart --hierLevels 3,3 --metricsDetail easy --numNodeWeights 2 --noRefinement --blockSizesFile testing/blockSizes_k9_w2.txt",
"mpirun -n 5 "+ ex + " --graphFile meshes/rotation-00000.graph --multiLevelRounds 5",
"mpirun -n 7 "+ ex + " --graphFile meshes/bubbles-00010.graph --localRefAlgo geographer",
"mpirun -n 12 "+ ex + " --graphFile meshes/rotation-00000.graph --initialPartition geoHierKM --hierLevels 2,3,2",
"mpirun -n 4 "+ ex + " --graphFile meshes/rotation-00000.graph --metricsDetail all --CGResidual 0.001",
"mpirun -n 4 "+ ex + " --graphFile meshes/rotation-00000.graph --metricsDetail all --maxCGIterations 50"
]


for cmd in allCommands:
    start_time = time.time()
    try:
        #subprocess.check_output(cmd, stderr=subprocess.STDOUT, stdin=subprocess.STDOUT, shell=True)
        output = subprocess.check_output(cmd, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
        print( cmd , end='')
        print(" :: Fail\n")
        exit(1)
    else:
        if verb:
            print("Output: \n{}".format(output))
        print( cmd , end='')
        print(" :: Success, run time %.2f\n"% (time.time() - start_time) )
