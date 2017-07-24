CXX = mpiCC -std=c++11 -ipo -O3 -no-prec-div -xHost -DNDEBUG

.PHONY: rebuild all clean
	
TARGETDIR = ../build
TARGETDIR_RANGECOMM = $(TARGETDIR)/RangeComm
TARGETDIR_COLL = $(TARGETDIR)/Coll
	
all : $(TARGETDIR) quicksort benchmark benchmark_elements
	
quicksort : $(TARGETDIR)/quicksort
benchmark : $(TARGETDIR)/benchmark
benchmark_elements : $(TARGETDIR)/benchmark_elements
	
QUICKSORT = $(TARGETDIR)/Main.o $(addprefix $(TARGETDIR_COLL)/,MPIOperations.o RangeOperations.o) \
	$(addprefix $(TARGETDIR_RANGECOMM)/,MPI_Ranged.o Barrier.o Bcast.o Gather.o Recv.o \
	Reduce.o Scan.o ScanAndBcast.o Send.o Sendrecv.o Functions.o)
$(TARGETDIR)/quicksort : $(QUICKSORT)
	$(CXX) -o $@ $(QUICKSORT)

# Create the executables
	
BENCHMARK = $(addprefix $(TARGETDIR_RANGECOMM)/,benchmark.o MPI_Ranged.o Barrier.o \
	Gather.o Bcast.o Recv.o Reduce.o Scan.o ScanAndBcast.o Send.o Sendrecv.o Functions.o)
$(TARGETDIR)/benchmark : $(BENCHMARK)
	$(CXX) -o $@ $(BENCHMARK)
BENCHMARK_E = $(addprefix $(TARGETDIR_RANGECOMM)/,benchmark_e.o MPI_Ranged.o Barrier.o Bcast.o \
	Gather.o Recv.o Reduce.o Scan.o ScanAndBcast.o Send.o Sendrecv.o Functions.o)
$(TARGETDIR)/benchmark_elements : $(BENCHMARK_E)
	$(CXX) -o $@ $(BENCHMARK_E)

Main.cpp = ../src/main.cpp
MAIN_HEADER = $(addprefix ../src/,io/QuickSortExecuter.hpp io/SortingDatatype.hpp \
	sort/tb_splitter.hpp sort/QuickSort.hpp CollectiveOperations/CollectiveOperations.hpp)
$(TARGETDIR)/Main.o : $(Main.cpp) $(MAIN_HEADER)
	$(CXX) -c -o $@ $(Main.cpp)
	
# Compile RangeComm source files
Functions.hpp = ../src/RangedComm/Functions.hpp
MPI_Ranged.cpp = ../src/RangedComm/MPI_Ranged.cpp
MPI_Ranged.hpp = ../src/RangedComm/MPI_Ranged.hpp ../src/RangedComm/Requests.hpp
$(TARGETDIR_RANGECOMM)/MPI_Ranged.o : $(MPI_Ranged.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(MPI_Ranged.cpp)
	
Barrier.cpp = ../src/RangedComm/Barrier.cpp
$(TARGETDIR_RANGECOMM)/Barrier.o : $(Barrier.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(Barrier.cpp)
	
Bcast.cpp = ../src/RangedComm/Bcast.cpp
$(TARGETDIR_RANGECOMM)/Bcast.o : $(Bcast.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(Bcast.cpp)

Gather.cpp = ../src/RangedComm/Gather.cpp
$(TARGETDIR_RANGECOMM)/Gather.o : $(Gather.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(Gather.cpp)
	
Recv.cpp = ../src/RangedComm/Recv.cpp
$(TARGETDIR_RANGECOMM)/Recv.o : $(Recv.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(Recv.cpp)

Reduce.cpp = ../src/RangedComm/Reduce.cpp
$(TARGETDIR_RANGECOMM)/Reduce.o : $(Reduce.cpp) $(MPI_Ranged.hpp) $(Functions.hpp)
	$(CXX) -c -o $@ $(Reduce.cpp)

Scan.cpp = ../src/RangedComm/Scan.cpp
$(TARGETDIR_RANGECOMM)/Scan.o : $(Scan.cpp) $(MPI_Ranged.hpp) $(Functions.hpp)
	$(CXX) -c -o $@ $(Scan.cpp)
	
ScanAndBcast.cpp = ../src/RangedComm/ScanAndBcast.cpp
$(TARGETDIR_RANGECOMM)/ScanAndBcast.o : $(ScanAndBcast.cpp) $(MPI_Ranged.hpp) $(Functions.hpp)
	$(CXX) -c -o $@ $(ScanAndBcast.cpp)
	
Send.cpp = ../src/RangedComm/Send.cpp
$(TARGETDIR_RANGECOMM)/Send.o : $(Send.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(Send.cpp)
	
Sendrecv.cpp = ../src/RangedComm/Sendrecv.cpp
$(TARGETDIR_RANGECOMM)/Sendrecv.o : $(Sendrecv.cpp) $(MPI_Ranged.hpp)
	$(CXX) -c -o $@ $(Sendrecv.cpp)
	
Functions.cpp = ../src/RangedComm/Functions.cpp
$(TARGETDIR_RANGECOMM)/Functions.o : $(Functions.cpp) $(Functions.hpp)
	$(CXX) -c -o $@ $(Functions.cpp)
	
benchmark.cpp = ../src/RangedComm/benchmark.cpp
$(TARGETDIR_RANGECOMM)/benchmark.o : $(benchmark.cpp)
	$(CXX) -c -o $@ $(benchmark.cpp)
$(TARGETDIR_RANGECOMM)/benchmark_e.o : $(benchmark.cpp)
	$(CXX) -D N_ELEMENTS -c -o $@ $(benchmark.cpp)

# Compile CollectiveOperations files
Coll.hpp = ../src/CollectiveOperations/CollectiveOperations.hpp

MPIOperations.cpp = ../src/CollectiveOperations/MPIOperations.cpp
MPIOP_HEADER = ../src/CollectiveOperations/MPIOperations.hpp $(Coll.hpp)
$(TARGETDIR_COLL)/MPIOperations.o : $(MPIOperations.cpp) $(MPIOP_HEADER)
	$(CXX) -c -o $@ $(MPIOperations.cpp)
	
RangeOperations.cpp = ../src/CollectiveOperations/RangeOperations.cpp
RANGE_HEADER = ../src/CollectiveOperations/MPIOperations.hpp $(Coll.hpp)
$(TARGETDIR_COLL)/RangeOperations.o : $(RangeOperations.cpp) $(RANGE_HEADER)
	$(CXX) -c -o $@ $(RangeOperations.cpp)
	
clean : 
	rm -rf $(TARGETDIR)

$(TARGETDIR) : $(TARGETDIR_RANGECOMM) $(TARGETDIR_COLL)
	mkdir -p $(TARGETDIR)
	
$(TARGETDIR_RANGECOMM) :
	mkdir -p $(TARGETDIR_RANGECOMM)/	
	
$(TARGETDIR_COLL) :
	mkdir -p $(TARGETDIR_COLL)/
	
rebuild : clean all
	
