#!/usr/bin/python3

import subprocess
import sys

subprocess.call("make -C build -j 12 install", shell=True)

allCommands = [ 
"mpirun -n 4 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph --noRefinement",
"mpirun -n 4 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph --metricsDetail all --maxCGIterations 500",
"mpirun -n 4 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph --metricsDetail all --CGResidual 0.001",
"mpirun -n 7 installation/bin/GeographerStandalone --graphFile meshes/slowrot-00000.graph --noRefinement",
"mpirun -n 16 installation/bin/GeographerStandalone --graphFile meshes/slowrot-00000.graph --noRefinement --blockSizesFile testing/blockSizes_2w.txt",
"mpirun -n 5 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph --multiLevelRounds 5",
"mpirun -n 7 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph --localRefAlgo parMetisRefine",
"mpirun -n 7 installation/bin/GeographerStandalone --graphFile meshes/bubbles-00010.graph --localRefAlgo geographer",
"mpirun -n 9 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph --noRefinement --autoSetCpuMem --processPerNode 3",
"mpirun -n 12 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph  --initialPartition geoHierKM --hierLevels 2,3,2",
"mpirun -n 11 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph  --initialPartition geoSFC --localRefAlgo parMetisRefine",
"mpirun -n 6 installation/bin/GeographerStandalone --graphFile meshes/rotation-00000.graph  --initialPartition geoSFC --localRefAlgo geographer --metricsDetail all"
]


for cmd in allCommands:
    try:
        #subprocess.check_output(cmd, stderr=subprocess.STDOUT, stdin=subprocess.STDOUT, shell=True)
        output = subprocess.check_output(cmd, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)
    else:
        #if sys.argv[1]=="-v":
        if len(sys.argv)>1:
            print("Output: \n{}".format(output))
        print( cmd , end='')
        print(" :: Success\n")
