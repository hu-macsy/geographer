Geographer
=========

Geographer is a mesh partitioner for large-scale simulation meshes in distributed memory. It partitions meshes in two steps: First with a fast geometric method, then with multi-level local refinement.

It is implemented in C++ and uses the LAMA framework for distributed graph datastructures. 

## Requirements
The following software is needed to install and use Geographer:

- [g++] (&gt;= 5.4)
- MPI
- A custom version of [Lama](https://github.com/kit-parco/lama)

The following libraries are necessary to build the unit tests:

- Boost (&gt;= 1.61.0)

## Installation

Geographer requires some custom additions to Lama for faster redistributions in distributed memory.
Please clone [our fork](https://github.com/kit-parco/lama), then compile and install the branch `looz-dmemo`.

After installing Lama and cloning this repository, enter the directory of this repository and call `cmake src -DSCAI_DIR=<path/to/lama>`, where `<path/to/lama>` is your Lama installation directory.

## Usage

Geographer is called from the command line.
By default, it is assumed that the input graph is in METIS format and that coordinate files describe one point position per line.
If no coordinate file is given, it is assumed that the coordinates are in foo.metis.xyz for an input file foo.metis.
For an input graph embedded in 2 dimensions, a graph file input.graph and a coordinate file input.graph.xyz, call the program like this:

    ./Geographer --graphFile input.graph  --numBlocks 8 --dimensions 2 --writePartition output.part

This will partition the graph into 8 blocks and write the result to output.part.

### Parallelization
Geographer can run in parallel using MPI. An example call with 8 processes :

    mpirun -np 8 Geographer --graphFile input.graph  --numBlocks 8 --dimensions 2 --writePartition output.part

The number of blocks can be different from the number of processes, but this will disable the local refinement. If the number of blocks is not given, it is set to the number of processes. Thus, the last command could also be phrased like this:

    mpirun -np 8 Geographer --graphFile input.graph --dimensions 2 --writePartition output.part

### Other parameters
Geographer supports other parameters and input formats as well. For a full list call `./Geographer --help`.
For example, to partition a graph formatted as METIS and coordinates given in the ADCIRC format into 512 blocks with a maximum imbalance of 0.01 according to the second node weight, use:

    mpirun -np 8 Geographer --graphFile fesom_core2.graph --coordFile node2d_core2.out --coordFormat OCEAN --nodeWeightIndex 1 --epsilon 0.01 --dimensions 2 --numBlocks 512

[g++]: https://gcc.gnu.org