CC=g++

FILES= ParcoRepartTest.cpp ParcoRepart.cpp
EXECUTABLE=Katanomi
LFLAGS= -L/home/moritzl/WAVE/scai_lama/install/lib -lscai_lama -lscai_dmemo -lscai_common -lscai_hmemo -lscai_utilskernel -L/home/moritzl/Includes/gtest-1.7.0/ -lgtest
CCFLAGS= --openmp --std=c++11 -I../../scai_lama/install/include/ -I/home/moritzl/Includes/gtest-1.7.0/include/ -DSCAI_TRACE_ON  -DSCAI_ASSERT_LEVEL_ERROR -DSCAI_LOG_LEVEL_ERROR

all:
	$(CC) $(CCFLAGS) $(FILES) -o $(EXECUTABLE) $(LFLAGS)

clean:
	rm -f $(EXECUTABLE)
