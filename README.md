Geographer
=========

Geographer is a mesh partitioner for large-scale simulation meshes in distributed memory. It partitions meshes in two steps: First with a fast geometric method, then with multi-level local refinement.

It is implemented in C++ and uses the LAMA framework for distributed graph datastructures. 

## Requirements
The following software is needed to compile and use libgeographer:

- A sufficiently modern compiler, for example [g++ &gt;= 4.9](https://gcc.gnu.org) or [icpc &gt;=17.0](https://en.wikipedia.org/wiki/Intel_C%2B%2B_Compiler)
- MPI
- The numerical library [Lama](https://github.com/kit-parco/lama) (&gt;= 3.0.0)
- For the command line frontend: [Boost](https://www.boost.org/) (&gt;= 1.61.0)
- For the unit tests: [Google Test](https://github.com/google/googletest)

## Installation

### Libraries
Compile and install the Lama library.
**You may need to do a `make install`
after cmake although the Lama website may state otherwise.**

The *RBC* library for splitting MPI communicators is included as a submodule in this repository.
If it is not yet cloned when starting the build process, cmake will download it automatically.
Should this fail, do it manually by calling `git submodule update --init --recursive`.
When compiling Geographer, RBC is automatically compiled with an external call to make which uses the compiler wrapper mpic++.

### Compilation
Create a build folder in the root directory of this repository, then in it call `cmake ..`.
Should you have installed Lama in a non-standard location, add `-DSCAI_DIR=<path/to/lama>` where `<path/to/lama>` is your Lama installation directory.
Afterwards, call `make` or `make Geographer` to create the executable.

## Usage as Standalone Executable
Geographer can be used as a library or called from the command line.
When using it from the command line, it expects to read an input graph from some file.
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

### Partitioning Methods

## Usage as Library

The methods that should be called by end users are the member functions of the ParcoRepart class. They mostly accept and return Lama data structures. The functions are templated to accept different types for indices and values, instantiated with the same types used in the compilation of the used Lama library.

An example is this:

	static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input,
		std::vector<DenseVector<ValueType>> &coordinates,
		struct Settings settings)

This function takes the input graph as an adjacency matrix in the CSRSparseMatrix format. The edge weights are interpreted as communication volumes. The coordinates are accepted in a vector of DenseVector, one DenseVector for each dimension. The settings struct contains relevant options, for example the target number of blocks. The result is returned as a DenseVector with one entry for each vertex, specifying its block ID in the partition. If the number of blocks is equal to the number of processes, the input data structures are redistributed to reflect the new partition.

### Node Weights

Node weights can also be given as an optional parameter. We accept multiple weights per node, for a graph with _n_ nodes and _k_ weights for each node, _nodeWeights_ should be a vector of _k_ DenseVectors with _n_ entries each.

	static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input,
		std::vector<DenseVector<ValueType>> &coordinates,
		std::vector<DenseVector<ValueType>> &nodeWeights,
		struct Settings settings)

### Metis Compatibility Interface

For those used to the common Metis library for graph partitioning, we offer an interface with a similar format:

	static std::vector<IndexType> partitionGraph(
		IndexType *vtxDist, IndexType *xadj, IndexType *adjncy, IndexType localM,
		IndexType *vwgt, IndexType ndims, ValueType *xyz,
		Settings  settings, Metrics& metrics)

### Mapping

