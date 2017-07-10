/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include <fstream>
#include <string>
#include <iostream>
#include <cmath>
#include "io/QuickSortExecuter.hpp"

void executeQuickSort(MPI_Comm comm, int seed, std::string datatype,
        long long elements, std::ofstream &file, int iterations) {
    SORT_TYPE type = SortingDatatype<void>::getSortType(datatype);
    switch (type) {
        case SORT_TYPE::Int:
        {
            QuickSortExecuter<int> qse(seed, elements, comm, file);
            qse.execute(iterations);
        }
            break;
        case SORT_TYPE::Long:
        {
            QuickSortExecuter<long> qse(seed, elements, comm, file);
            qse.execute(iterations);
        }
            break;
        case SORT_TYPE::Double:
        {
            QuickSortExecuter<double> qse(seed, elements, comm, file);
            qse.execute(iterations);
        }
            break;
        case SORT_TYPE::Float:
        {
            QuickSortExecuter<float> qse(seed, elements, comm, file);
            qse.execute(iterations);
        }
            break;
    }
}

/*
 * arguments: log of starting number of elements per PE, datatype, seed
 */
int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << MPI_Wtick() << std::endl;
    int seed = 3242359;
    std::string datatype = "double"; 
    int max = 20;
    if (argc > 1)
        max = atoi(argv[1]);
    if (argc > 2)
        std::string datatype = argv[2];
    if (argc > 3)
	seed = atoi(argv[3]);

    std::ofstream file, tmp_file;
    if (rank == 0) {
        file.open("Quicksort.txt", std::ios::app);
	tmp_file.open("tmp.txt", std::ios::app);
    }    

    executeQuickSort(MPI_COMM_WORLD, seed, datatype, pow(2, 15), tmp_file, 10);

    int iterations = 7;
    for (int n = 0; n <= max; n++) {
	long long elements = pow(2, n);
        executeQuickSort(MPI_COMM_WORLD, seed, datatype, elements, file, iterations);
    }
    if (rank == 0){
        file.close();
	tmp_file.close();
    }
    double end_time = MPI_Wtime();
    if (rank == 0)
        std::cout << "Total time: " << end_time - start_time << std::endl;
    // Finalize the MPI environment.
    MPI_Finalize();
}
