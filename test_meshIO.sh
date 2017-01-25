#!/bin/bash

for run in {0..10}
do
  	mpirun -np 2 Plegma --gtest_filter=*Struct*
	#./a.out
done
