@file
@mainpage User guide

Geographer
=========

Geographer is a mesh partitioner for large-scale simulation meshes in distributed memory. It partitions meshes in two steps: First with a fast geometric method, then with multi-level local refinement.

It is implemented in C++ and uses the LAMA framework for distributed graph datastructures. 

## Requirements
The following software is needed to compile and use Geographer:

- A sufficiently modern compiler, for example [g++ &gt;= 4.9](https://gcc.gnu.org) or [icpc &gt;=17.0](https://en.wikipedia.org/wiki/Intel_C%2B%2B_Compiler). LLVM / Clang is currently not supported.
- MPI
- OpenMP
- CMake (&gt;= 3.0.2)
- The numerical library [Lama](https://github.com/hu-macsy/lama) (&gt;= 3.0.0)
- For Lama, an implementation of the BLAS linear algebra standard

On a recent Ubuntu (&gt;= 16.04), most of the dependencies can be installed with the following command:

	sudo apt install cmake g++ git libatlas-base-dev mpi-default-dev


## Installation

### Libraries
Compile and install the Lama library.
If its dependencies are installed, this can be done with the following commands:

    git clone https://github.com/hu-macsy/lama.git
    mkdir lama/build && cd lama/build && cmake ../scai && cd ../..
    cd lama/build && make && sudo make install && cd ../..

If root access is not available or not preferred, you can pass the argument `-DCMAKE_INSTALL_PREFIX=<path>`to cmake to specify an alternative install location.

Note that on newer versions of GCC (&gt;= 8), several type conversions trigger warnings which are interpreted as errors.
As a workaround, give `-DADDITIONAL_WARNING_FLAGS=""` as an argument to cmake.

The *KaDIS* library for distributed sorting is included as a submodule in this repository.
If it is not yet cloned when starting the build process, cmake will download it automatically.
Should this fail, do it manually by calling `git submodule update --init --recursive`.
When compiling Geographer, KaDIS is automatically compiled as a CMake subproject.

### Compilation
Use CMake to configure the project and make to build it, followed by make install:

	mkdir build && cd build && cmake .. && make && sudo make install

If you have installed Lama in a non-standard location, add `-DSCAI_DIR=<path/to/lama>` where `<path/to/lama>` is your Lama installation directory.
To install Geographer in an alternative location, pass the argument `-DCMAKE_INSTALL_PREFIX=<path>` to cmake. After successful compilation, the library `libgeographer` and the standalone executable `GeographerStandalone` are installed into the installation target. If the Google Test library was found, the unit tests can be found in the executable `GeographerTest`.

### Mac OS
On Mac OS, the default compiler is LLVM, which is not supported by the Lama library on which we depend.
One possibility to install the Gnu Compiler Collection GCC is with the [homebrew](https://brew.sh/) package manager.
You need to specify the compiler explicitly when calling CMake;
 as on Mac OS the command `g++` is often an alias for clang, you might need to set the full path, for example as `-DCMAKE_CXX_Compiler=/usr/local/bin/g++-7 -DCMAKE_C_COMPILER=/usr/local/bin/gcc-7`.

## Usage as Standalone Executable
Geographer can be used as a library or called from the command line.
When using it from the command line, it expects to read an input graph from a file.
By default, it is assumed that the input graph is in the METIS format and that coordinate files describe one point position per line.
If no coordinate file is given, it is assumed that the coordinates are in foo.metis.xyz for an input file foo.metis.
For an input graph embedded in 2 dimensions, a graph file input.graph and a coordinate file input.graph.xyz, call the program like this:

    ./GeographerStandalone --graphFile input.graph  --numBlocks 8 --dimensions 2 --outFile output.part

This will partition the graph into 8 blocks and write the result to output.part.

### Parallelization
Geographer can run in parallel using MPI. An example call with 8 processes :

    mpirun -np 8 GeographerStandalone --graphFile input.graph  --numBlocks 8 --dimensions 2 --outFile  output.part

The number of blocks can be different from the number of processes, but this will disable the local refinement. If the number of blocks is not given, it is set to the number of processes. Thus, the last command could also be phrased like this:

    mpirun -np 8 GeographerStandalone --graphFile input.graph --dimensions 2 --outFile output.part

### Other parameters
Geographer supports other parameters and input formats as well. For a full list call `./GeographerStandalone --help`.
For example, to partition a graph formatted as METIS and coordinates given in the ADCIRC format into 512 blocks with a maximum imbalance of 0.01 according to the second node weight, use:

    mpirun -np 8 GeographerStandalone --graphFile fesom_core2.graph --coordFile node2d_core2.out --coordFormat ADCIRC --epsilon 0.01 --dimensions 2 --numBlocks 512

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

For those who are used to the common Metis library for graph partitioning, we offer an interface with a similar format:

	static std::vector<IndexType> partitionGraph(
		IndexType *vtxDist, IndexType *xadj, IndexType *adjncy, IndexType localM,
		IndexType *vwgt, IndexType ndims, ValueType *xyz,
		Settings  settings, Metrics& metrics)

The _Metrics_ struct keeps track of how long each of the partitioning substeps took, as well as other performance metrics. It can be constructed with a settings object.

	Settings settings;
	settings.numBlocks = 8 //example, setting the target number of blocks to 8.
	Metrics metrics(settings);

### Mapping

The performance of a distributed application depends not only on balancing the computational load and reducing the amount of necessary communication, but also on a good match between the partitioned blocks and the available computing resources. This holds especially for heterogenuous environments with for example a mix of GPUs and CPUs.

If you know the topology of your computing system, you can pass a representation of it as the _commTree_ argument to the partitionGraph method.

	static DenseVector<IndexType> partitionGraph(
		CSRSparseMatrix<ValueType> &input,
		std::vector<DenseVector<ValueType>> &coordinates,
		std::vector<DenseVector<ValueType>> &nodeWeights,
		DenseVector<IndexType>& previous,
		CommTree<IndexType,ValueType> commTree,
		struct Settings settings,
		struct Metrics& metrics)

_TODO_: Describe how to construct a comm tree. This is also not optimal now.

## Partitioning Methods

The partitioning works in two phases: In the geometric phase, the input points are partitioned to yield a convex partition, but without considering the graph-based metrics.
In the following graph phase, if the number of blocks equals the number of processes, use the multi-level heuristic with distributed local refinement to improve the solution with respect to graph-theoretical metrics such as the cut and communication volume.

The choice of method in the geometric phase is governed by the parameter _initialPartition_ in the _Settings_ struct or the --initialPartition argument of the standalone executable.
Several methods are available: 

### Space Filling Curves

Sort and redistribute points along a Hilbert curve in two or three dimensions. Fast, but with suboptimal quality. Does not support node weights. Selected by setting the _initialPartition_ member of settings to _ITI::Tool::geoSFC_ or passing "--initialPartition geoSFC" to the standalone executable.

### K-Means

First use space-filling curves for an initial partition, then improve it using distributed k-means. Repeatedly adjust influence values to achieve the desired balance. Output partitions are convex or nearly convex. This is the default, represented by _ITI::Tool::geoKmeans_ or "--initialPartition geoKMeans".

### Hierarchical K-Means

For more than a few thousand blocks, k-means sometimes takes long to converge to a balanced solution. A faster alternative is to first partition into a smaller number of blocks and then proceed hierarchically, partitioning each block further until the desired number of blocks is reached.
To use this method, select _ITI::Tool::geoHierKM_ or "--initialPartition geoHierKM".
You also need to pass the number of divisions on each level in the _hierLevels_ argument.
The final number of blocks is the product of the divisions on all levels, the following snippet sets 4 levels of 100 blocks in total:

	ITI::Settings settings;
	settings.numBlocks = 100;
	settings.initialPartition = ITI::Tool::geoHierKM;
	settings.hierLevels = std::vector<int>({5, 2, 5, 2});

### MultiSection

Analogous to recursive bisection, _MultiSection_ repeatedly divides the initial point set along a set of straight lines. It is selected with _ITI::Tool::geoMS_ or "--initialPartition geoMS".
When using this method, you also need to pass the number of cuts in each dimension with the cutsPerDim parameter. The final number of blocks is product of the number of cuts in each dimension. The following snipped sets 100 blocks in total in two dimensions:

	ITI::Settings settings;
	settings.numBlocks = 100;
	settings.dimensions = 2;
	settings.initialPartition = ITI::Tool::geoMS;
	settings.hierLevels = std::vector<int>({10, 10});

The number of dimensions must match the number of dimensions in the input.
_TODO_: give credit to original authors

### Multilevel Local Refinement

The graph refinement phase uses the Fiduccia-Mattheyses method to improve the geometric partition. It is only available if the number of blocks matches the number of processes. It can be disabled by setting settings.noRefinement to False or with the --noRefinement flag when using the standalone executable.
