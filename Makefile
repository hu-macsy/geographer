CC=g++

FILES= ParcoRepartTest.cpp ParcoRepart.cpp
EXECUTABLE=Katanomi
LFLAGS= -L/home/moritzl/WAVE/scai_lama/install/lib -lscai_lama -lscai_dmemo -lscai_common -lscai_hmemo -L/home/moritzl/Includes/gtest-1.7.0/ -lgtest
CCFLAGS= --openmp --std=c++11 -I../../scai_lama/install/include/ -DSCAI_TRACE_ON -I/home/moritzl/Includes/gtest-1.7.0/include/

all:
	$(CC) $(CCFLAGS) $(FILES) -o $(EXECUTABLE) $(LFLAGS)

clean:
	rm -f $(EXECUTABLE)
