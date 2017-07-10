/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef QUICKSORTEXECUTER_HPP
#define QUICKSORTEXECUTER_HPP

#include "../RangedComm/MPI_Ranged.hpp"
#include "../sort/QuickSort.hpp"
#include <mpi.h>
#include <fstream>
#include <random>
#include <cmath>

template<typename T>
class QuickSortExecuter {
public:

    QuickSortExecuter(int seed, long long elements, MPI_Comm comm, std::ofstream &file) :
    elements(elements), seed(seed), comm(comm), file(&file) {
        unsorted_data_uni = new T[elements];
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        generator.seed(seed + rank);
        createData(unsorted_data_uni, elements);
    }

    ~QuickSortExecuter() {
        delete[] unsorted_data_uni;
    }

    void createData(T *data, int split_size);

    void execute(int iterations) {
        std::vector<std::string> pivots{"9", "16_log_p", "1/50_s"};

//        for (int i = 0; i < pivots.size(); i++)
//            benchmark("range", iterations, pivots[i]);
//        benchmark("range", iterations, "16_50", true);
//        benchmark("range", iterations, "16_50", false, true);
//        benchmark("range", iterations, "16_50", false, false, true);

        benchmark("range", iterations, "16_50");
    }

    void benchmark(std::string collective, int iterations, std::string pivot,
            bool blocking = false, bool mpi_split = false, bool barriers = false) {
        for (int iteration = 0; iteration < iterations; iteration++) {
            T* data = new T[elements];
            std::string data_dist;
            
            data_dist = "uniform";
            for (int i = 0; i < elements; i++)
                data[i] = unsorted_data_uni[i];
            
            std::string datatype = SortingDatatype<T>::getString();

            std::vector<double> timers, max_timers, avg_timers, min_timers;
            std::vector<std::string> timer_names;
            if (collective == "mpi") {
		mpi_split = true;
                QuickSort<T, MPI_Comm, MPI_Request> qs(blocking, mpi_split, barriers, pivot);
                MPI_Barrier(comm);
                qs.sort(data, elements, comm, seed);
                qs.getTimers(timer_names, timers, max_timers, avg_timers, min_timers, comm);
            } else if (collective == "range") {
                QuickSort<T, Range::Comm, Range::Request> qs(blocking, mpi_split, barriers, pivot);
                MPI_Barrier(comm);
                qs.sort(data, elements, comm, seed);
                qs.getTimers(timer_names, timers, max_timers, avg_timers, min_timers, comm);
            }
            
//            double timer_array[size * timers.size()];
//            MPI_Gather(&timers[0], timers.size(), MPI_DOUBLE, timer_array, 
//                    timers.size(), MPI_DOUBLE, 0, comm);
//            double exchange_timer_array[size * exchange_timers.size()];
//            MPI_Gather(&exchange_timers[0], exchange_timers.size(), MPI_DOUBLE, 
//                    exchange_timer_array, exchange_timers.size(), MPI_DOUBLE, 0, comm);
//            double exchange_split_array[size * exchange_split.size()];
//            MPI_Gather(&exchange_split[0], exchange_split.size(), MPI_DOUBLE, 
//                    exchange_split_array, exchange_split.size(), MPI_DOUBLE, 0, comm);
            
/*            double max_exchange = 0.0;
	    int max_PE = 0;
            for (int l = 0; l < timer_names.size(); l++) {
                if (timer_names[l] == "exchange") {
                    for (int j = 0; j < size; j++) {
                        double time = timer_array[j * timers.size() + l];
                        if (time > max_exchange) {
                            max_PE = j;
                            max_exchange = time;
                        }
                    }
                }                
            }*/ 
//            double *max_exchange_timers = &exchange_timer_array[max_PE * exchange_timers.size()];
//            double *max_exchange_split = &exchange_split_array[max_PE * exchange_split.size()];

//            bool isSort = isSorted(data, elements, comm);  
//            if (!isSort)
//                std::exit(12);

            if (rank == 0) {

                double sum_max = 0.0;
                int timer = 0;
                if (barriers) {
                    while (timer_names[timer] != "sum") {
                        sum_max += max_timers[timer];
                        timer++;
                    }
                    max_timers[timer] = sum_max;
                }
                    
                std::string result = "RESULT size=" + std::to_string(size) 
                        + " collective=" + collective
                        + " datatype=" + datatype
                        + " distribution=" + data_dist
                        + " elements=" + std::to_string(elements)
                        + " iteration=" + std::to_string(iteration)
                        + " blocking=" + std::to_string(blocking)
                        + " mpi_split=" + std::to_string(mpi_split)
                        + " barriers=" + std::to_string(barriers)
                        + " pivot=" + pivot;
                
                std::cout << "RESULT " << std::endl
                        << "Collective: " << collective << std::endl
                        << "P:          " << size << std::endl
                        << "N/P:        " << elements << std::endl
                        << "Type:       " << datatype << std::endl
                        << "Data Distr: " << data_dist << std::endl
                        << "Pivot     : " << pivot << std::endl
                        // << "seed=      " << seed << std::endl
//                        << "isSorted:   " << (isSort ? "Yes" : "No") << std::endl
                        << "blocking:   " << blocking << std::endl
                        << "mpi_split:  " << mpi_split
			<< std::endl;

                std::cout << std::endl;        
                for (int n = 0; n < timers.size(); n++) {
                    std::cout << timer_names[n] << ": " << min_timers[n] << ", " << avg_timers[n]
			<< ", " << max_timers[n] << std::endl;
                    result += " " + timer_names[n] + "=" + std::to_string(max_timers[n]);
                }
/*                std::cout << "exchange split:" << std::endl << "    ";
                for (int n = 0; n < 4; n++) {
                    std::cout << max_exchange_split[n] << ", ";
                }
                std::cout << std::endl;
                std::cout << "exchange times: [min avg max]";
                for (int n = 0; n < exchange_timers_max.size() && exchange_timers_max[n] > 0.0; n++) {
	            std::cout << std::endl << "   ";
		    std::cout << "[" << exchange_timers_min[n] << " | " << exchange_timers_avg[n] 
                            << " | " << exchange_timers_max[n] << "]";
                }
                std::cout << std::endl;*/
                result += "\n";
                (*file) << result;
                std::cout << "---------------------------" << std::endl;
            }
            delete[] data;
        }
    }

private:

    bool isSorted(T *data, int N, MPI_Comm comm) {
        MPI_Datatype mpi_type = SortingDatatype<T>::getMPIDatatype();
        if (rank < size - 1) {
            MPI_Send(&data[N - 1], 1, mpi_type, rank + 1, 10, comm);
        }
        if (rank > 0) {
            T last_left;
            MPI_Recv(&last_left, 1, mpi_type, rank - 1, 10, comm, MPI_STATUS_IGNORE);
            if (last_left > data[0])
                return false;
        }
        for (int i = 1; i < N; i++) {
            if (data[i - 1] > data[i])
                return false;
        }
        return true;
    }

    long long elements;
    int seed, size, rank;
    MPI_Comm comm;
    std::ofstream *file;
    std::mt19937 generator;
    T *unsorted_data_uni, *unsorted_data_bad;
};

template<>
void QuickSortExecuter<int>::createData(int *data, int split_size) {
    int min = std::numeric_limits<int>::min();
    int max = std::numeric_limits<int>::max();
    std::uniform_int_distribution<int> distribution(min, max);

    for (int i = 0; i < split_size; i++) {
        data[i] = distribution(generator);
    }
}

template<>
void QuickSortExecuter<long>::createData(long *data, int split_size) {
    long min = std::numeric_limits<long>::min();
    long max = std::numeric_limits<long>::max();
    std::uniform_int_distribution<long> distribution(min, max);

    for (int i = 0; i < split_size; i++) {
        data[i] = distribution(generator);
    }
}

template<>
void QuickSortExecuter<double>::createData(double *data, int split_size) {
    double min = 0.0;
    double max = 100.0; //std::numeric_limits<double>::max();
    std::uniform_real_distribution<double> distribution_uni(min, max);

    for (int i = 0; i < split_size; i++) {
        data[i] = distribution_uni(generator);
    }
    
/*    int group = 128, col_size = split_size, nprocs = size, myrank = rank;
    
    std::uniform_int_distribution<int> distr(0, std::numeric_limits<int>::max());

    int i, j, k, v;

    double multiplier, offset, t1, t2;

    offset = std::numeric_limits<int>::max() / 2.0;
    multiplier = std::numeric_limits<double>::max() / offset;
    t1 = std::numeric_limits<int>::max() / ((double) (int) nprocs);
    v = (((int) myrank - ((int) myrank % group)) + ((int) nprocs / 2)) %
            (int) nprocs;
    
    k = 0;
    for (i = 0; i < group; i++) {
        t2 = (((double) v) * t1) - offset;
        for (j = 0; j < (col_size / group); j++)
            data[k++] = (t2 + (((double) distr(generator)) / ((double) nprocs))) * multiplier;
        v = (v + 1) % (int) nprocs;
    }*/
}

template<>
void QuickSortExecuter<float>::createData(float *data, int split_size) {
    float min = 0.0;
    float max = std::numeric_limits<float>::max();
    std::uniform_real_distribution<float> distribution(min, max);

    for (int i = 0; i < split_size; i++) {
        data[i] = distribution(generator);
    }
}

#endif /* QUICKSORTEXECUTER_HPP */

